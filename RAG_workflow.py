# Standard library imports
import os
import sys
import time
import re
import json
import gzip
import csv
import hashlib
import sqlite3
import argparse
import warnings
import atexit
import functools
import logging
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Third-party imports
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv

import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.data import find

from PyPDF2 import PdfReader
from transformers import AutoModel, AutoTokenizer

import faiss
from rank_bm25 import BM25Okapi, BM25Plus

from openai import OpenAI
import google.generativeai as genai
from google.generativeai import types
from anthropic import Anthropic
import torch
from openai import OpenAI as DeepSeekClient

# Environment & API clients setup
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client_open_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_gemini = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
client_claude = OpenAI(
    api_key=os.getenv("CLAUDE_API_KEY"),
    base_url="https://api.anthropic.com/v1/"
)
client_deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
client_grok = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.getenv("XAI_API_KEY"),
)

# NLTK: silent downloader
logging.getLogger('nltk').setLevel(logging.ERROR)


def ensure_nltk_resource(pkg_name: str, resource_path: str) -> None:
    """
    Check for an NLTK resource; if missing, download quietly
    with no stdout/stderr noise.
    """
    try:
        find(resource_path)
    except LookupError:
        with contextlib.redirect_stdout(open(os.devnull, 'w')), \
                contextlib.redirect_stderr(open(os.devnull, 'w')):
            nltk.download(pkg_name, quiet=True)


ensure_nltk_resource('punkt',
                     'tokenizers/punkt')
ensure_nltk_resource('stopwords', 'corpora/stopwords')


class ConfigError(Exception):
    pass


def load_config(path: str, print_settings: Optional[bool] = False) -> Dict[str, Any]:
    """
    Load JSON config from `path`, enforce required keys,
    fill in defaults, print each key (excluding secrets) with its source,
    then return the merged dict.
    """
    default_cfg: Dict[str, Any] = {
        "number_of_expansions": 5,
        "batch_size": 64,
        "embeddings_model_name": "mghuibregtse/biolinkbert-large-simcse-rat",
        "generation_model": "o4-mini",
        "validation_model": "grok-3-mini",
        "amount_docs": 50,
        "weight_faiss": 50,
        "weight_bm25": 50,
        "fdr_threshold": 0.05,
        "query_range": 1,
    }

    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")

    def escape_newlines(m: re.Match) -> str:
        inner = m.group(0)[1:-1].replace("\n", "\\n")
        return f"\"{inner}\""

    processed = re.sub(r'"(?:\\.|[^"\\])*"', escape_newlines, raw, flags=re.DOTALL)

    try:
        user_cfg = json.loads(processed)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path!r}: {e}")

    legacy_aliases = {
        "model_name": "embeddings_model_name",
        "model": "embeddings_model_name",
    }
    for legacy_key, current_key in legacy_aliases.items():
        if legacy_key in user_cfg and current_key not in user_cfg:
            user_cfg[current_key] = user_cfg[legacy_key]

    # Required keys
    for key in ("system_instruction_response", "query"):
        if key not in user_cfg:
            raise ConfigError(f"Missing required config key: {key}")

    merged = default_cfg.copy()
    merged.update(user_cfg)

    missing_opt = set(default_cfg) - set(user_cfg)
    if missing_opt:
        logging.warning("Using default for: %s", ", ".join(sorted(missing_opt)))

    print(f"Using config: {Path(path).name} with these parameters:")
    if print_settings:
        for k, v in merged.items():
            if k in ("query", "system_instruction_response"):
                continue
            src = "user" if k in user_cfg else "default"
            print(f"{k}: {v} (from {src} config)")

    return merged


# BM25 stop-words setup
stop_words = set(stopwords.words('english'))
stop_words |= {
    "list", "genes", "identify", "mechanisms", "pathways",
    "related", "based", "following", "recent", "study"
}

os.makedirs("./logs", exist_ok=True)


def compute_file_hash(file_path: str, block_size: int = 65536) -> str:
    """
    Computes the SHA-256 has of a file's content.

    Args:
        file_path: The path to the file to hash.
        block_size: The size (in bytes) of each block to hash.

    Returns:
        A hexadecimal string representing the SHA-256 hash of the file.

    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as hash_file:
        for block in iter(lambda: hash_file.read(block_size), b''):
            hasher.update(block)
    return hasher.hexdigest()


def initialize_gene_list(
        gene_file_path: str = r"./data/GSEA/genes_of_interest/C3_9w_detab_fixed.txt",
        fdr_threshold: Optional[float] = None,
    de_filter_option: str = "combined",
    excel_file_path: Optional[str] = None,
) -> Tuple[str, str, int]:
    """
    Creates a list of genes from a plain-text file (preferred) or Excel file (legacy).

    Args:
        gene_file_path: Path to a gene list file.
            - .txt: one gene symbol per line (no filtering is applied)
            - .xlsx/.xls: legacy mode using DE/FDR filtering
        fdr_threshold: Legacy FDR threshold for Excel mode.
        de_filter_option: Legacy DE filtering mode for Excel mode.

    Returns:
        A tuple containing:
            - gene_list_string: A comma‐separated string of selected gene names.
            - regulation: "provided_list" for txt mode, or legacy regulation label for Excel mode.
            - num_genes: The number of genes included in the final list.
    """
    if excel_file_path:
        gene_file_path = resolve_gene_excel_path(excel_file_path)

    if gene_file_path.lower().endswith('.txt'):
        if not os.path.exists(gene_file_path):
            raise FileNotFoundError(f"Gene list file not found: {gene_file_path}")

        genes: List[str] = []
        seen: Set[str] = set()
        with open(gene_file_path, 'r', encoding='utf-8') as gene_file:
            for line in gene_file:
                gene = line.strip()
                if not gene or gene in seen:
                    continue
                seen.add(gene)
                genes.append(gene)

        return ", ".join(genes), "provided_list", len(genes)

    if fdr_threshold is None:
        fdr_threshold = 0.05

    results = process_excel_data(gene_file_path, de_filter_option, fdr_threshold)
    if results:
        gene_list_string, regulation, num_genes = results[0]
    else:
        gene_list_string = ""
        regulation = ""
        num_genes = 0
    return gene_list_string, regulation, num_genes


def process_excel_data(
        path: str,
        mode: str,
        fdr: float
) -> List[Tuple[str, str, int]]:
    """
    Processes an Excel file to filter and extract gene data based on differential expression and FDR threshold.

    Args:

        path: The file path to the Excel file containing gene data.
        mode: The differential expression filter option ('combined' or 'separate').
        fdr: False discovery rate threshold for filtering.

    Returns:
        A list of tuples. Each tuple contains:
            - A comma-separated string of gene names.
            - A regulation label ('upregulated', 'downregulated', or 'combined').
            - The number of genes included in that subset.
    """
    df = pd.read_excel(path)
    out: List[Tuple[str, str, int]] = []

    def collect(sub_df: pd.DataFrame, regulation_label: str):
        genes = sub_df['X'].tolist()
        out.append((', '.join(genes), regulation_label, len(genes)))

    if mode == "combined":
        df = df[df.DE != 0]
        if not df.empty and df.FDR.max() <= fdr:
            unique = set(df.DE)
            regulation = "upregulated" if unique == {1} else "downregulated" if unique == {-1} else "combined"
            collect(df, regulation)

    elif mode == "separate":
        for val, label in ((1, "upregulated"), (-1, "downregulated")):
            sub = df[df.DE == val]
            if not sub.empty and sub.FDR.max() <= fdr:
                collect(sub, label)

    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'combined' or 'separate'")

    return out


def resolve_gene_excel_path(
        preferred_path: Optional[str] = None,
        genes_dir: str = r"./data/GSEA/genes_of_interest"
) -> str:
    """
    Resolve the Excel file path used for gene initialization.

    Priority:
      1) user-provided preferred_path if it exists,
      2) first .xlsx file found in genes_dir (sorted),
      3) legacy default path.
    """
    legacy_default = r"data/GSEA/genes_of_interest/PMP22_VS_WT.xlsx"
    requested = preferred_path or legacy_default

    if os.path.exists(requested):
        return requested

    if os.path.isdir(genes_dir):
        excel_files = sorted(
            [f for f in os.listdir(genes_dir) if f.lower().endswith(".xlsx")]
        )
        if excel_files:
            fallback_path = os.path.join(genes_dir, excel_files[0])
            print(
                f"Requested Excel file not found: {requested}. "
                f"Using fallback: {fallback_path}"
            )
            return fallback_path

    return requested


def load_gene_list_from_descriptions(
    csv_path: str = r"./data/GSEA/external_gene_data/gene_descriptions.csv",
) -> Tuple[str, int]:
    """
    Load gene names from gene_descriptions.csv and return a comma-separated list plus count.

    Args:
        csv_path: Path to gene_descriptions.csv containing at least a 'Gene name' column.
    Returns:
        Tuple of (gene_list_string, num_genes).
    """
    if not os.path.exists(csv_path):
        return "", 0

    try:
        df = pd.read_csv(csv_path, usecols=["Gene name"])
    except Exception as e:
        logging.warning("Failed to load genes from %s: %s", csv_path, e)
        return "", 0

    genes = [g.strip() for g in df["Gene name"].dropna().astype(str).tolist() if g.strip()]
    genes = list(dict.fromkeys(genes))
    return ", ".join(genes), len(genes)


def resolve_gene_list_file(
        preferred_path: Optional[str] = None,
        genes_dir: str = r"./data/GSEA/genes_of_interest"
) -> str:
    """Resolve a usable gene list file path without requiring hardcoded config values."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    if preferred_path and not os.path.exists(preferred_path):
        print(f"Configured gene list file not found: {preferred_path}. Searching fallback in {genes_dir}.")

    if os.path.isdir(genes_dir):
        txt_files = sorted([f for f in os.listdir(genes_dir) if f.lower().endswith(".txt")])
        if txt_files:
            return os.path.join(genes_dir, txt_files[0])

        excel_files = sorted(
            [f for f in os.listdir(genes_dir) if f.lower().endswith((".xlsx", ".xls"))]
        )
        if excel_files:
            return os.path.join(genes_dir, excel_files[0])

    raise FileNotFoundError(
        f"No gene list file found. Set 'gene_list_path' in config or add a .txt/.xlsx file under {genes_dir}."
    )


def load_gene_id_cache(file_path: str) -> Dict[str, str]:
    """
    Loads the gene ID cache from a JSON file.

    Args:
        file_path: The path to the JSON file containing the gene ID cache.

    Returns:
        A dictionary with the gene ID cache, or an empty dictionary if the file does not exist or cannot be decoded.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as gene_id_file:
            try:
                return json.load(gene_id_file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file: {e}")
                return {}
    else:
        return {}


def save_gene_id_cache(
        cache: Dict[str, str],
        file_path: str
) -> None:
    """
    Saves the gene ID cache to a JSON file.

    Args:
        cache: A dictionary mapping gene IDs (as strings) to their resolved gene symbols.
        file_path: The path to the JSON file where the cache should be saved.

    Returns:
        None. Writes the cache dictionary to disk in JSON format.
    """
    print(f"\n\n\nSaving cache to {file_path}\n\n")
    with open(file_path, 'w', encoding='utf-8') as gene_id_file:
        json.dump(cache, gene_id_file, indent=4)


def search_genes(
        unknown_genes: Set[str],
        gene_cache: Dict[str, str],
        cache_file: str
) -> Dict[str, str]:
    """
    Searches for gene symbols for a set of unknown genes using the MyGene.info API.
    Updates the gene cache with newly resolved symbols and saves the updated cache.

    Args:
        unknown_genes: A list of gene IDs (strings) that require symbol resolution.
        gene_cache: A dictionary containing already resolved gene ID → symbol mappings.
        cache_file: The path to the JSON file where the updated cache will be saved.

    Returns:
        A dictionary mapping each original gene ID (str) to its resolved gene symbol (str).
    """
    url = "https://mygene.info/v3/gene/"
    resolved_genes = {}

    for gene_id in unknown_genes:
        sanitized_gene_id = gene_id.strip("[]")
        print(f"Searching for gene symbol for {sanitized_gene_id}...")
        try:
            response = requests.get(url + sanitized_gene_id)
            response.raise_for_status()
            data = response.json()

            if 'symbol' in data:
                gene_symbol = data['symbol']
                resolved_genes[sanitized_gene_id] = gene_symbol
                print(f"Found gene symbol for {sanitized_gene_id}: {gene_symbol}")
            else:
                resolved_genes[sanitized_gene_id] = "Unknown"
        except requests.exceptions.RequestException as e:
            resolved_genes[sanitized_gene_id] = "Error"
            print(f"Error with gene ID {sanitized_gene_id}: {e}")

    gene_cache.update(resolved_genes)
    save_gene_id_cache(gene_cache, cache_file)

    return resolved_genes


def convert_gene_id_to_symbols(
        file: str,
        data_dir: str,
        ncbi_json_dir: str
) -> Tuple[List[str], List[str]]:
    """
    Converts gene IDs in a file to gene symbols using a cached mapping and by searching unknown genes via an API.
    If the file is compressed, it is decompressed before processing.

    Args:
        file: The filename to process.
        data_dir: The directory containing the file.
        ncbi_json_dir: The directory where the gene ID cache is stored.

    Returns:
        A tuple containing:
            - A list of processed lines with gene symbols.
            - A list of gene IDs that remain unknown.
    """
    cache_file = os.path.join(ncbi_json_dir, 'ncbi_id_to_symbol.json')
    gene_cache = load_gene_id_cache(cache_file)
    output_lines = []
    unknown_genes = set()
    sanitized_gene_id_map = {}

    if file.endswith('.gz'):
        decompressed_file = file[:-3]  # Remove '.gz' extension
        with gzip.open(os.path.join(data_dir, file), 'rt') as gzfile:
            with open(os.path.join(data_dir, decompressed_file), 'w') as outfile:
                for line in gzfile:
                    outfile.write(line)
        file = decompressed_file

    with open(os.path.join(data_dir, file), 'r') as infile:
        for line_index, line in enumerate(infile):
            parts = line.strip().split('\t')
            pathway_info = parts[:2]
            gene_ids = parts[2:]
            gene_symbols = []

            for i, gene_id in enumerate(gene_ids):
                sanitized_gene_id = gene_id.strip("[]")
                if sanitized_gene_id in gene_cache:
                    gene_symbols.append(gene_cache[sanitized_gene_id])
                else:
                    if sanitized_gene_id.isdigit():  # Makes sure that they aren't already converted
                        gene_symbols.append("Unknown")
                        unknown_genes.add(sanitized_gene_id)

                        if line_index not in sanitized_gene_id_map:
                            sanitized_gene_id_map[line_index] = []
                        sanitized_gene_id_map[line_index].append((i, sanitized_gene_id))
                    else:
                        gene_symbols.append(sanitized_gene_id)

            new_line = '\t'.join(pathway_info + gene_symbols)
            output_lines.append(new_line)

    if unknown_genes:
        print(f"There are unknown genes: {unknown_genes}")
        resolved_genes = search_genes(unknown_genes, gene_cache, cache_file)

        for line_index, unknown_gene_positions in sanitized_gene_id_map.items():
            line_parts = output_lines[line_index].split('\t')
            pathway_info = line_parts[:2]
            gene_symbols = line_parts[2:]

            for position, sanitized_gene_id in unknown_gene_positions:
                if sanitized_gene_id in resolved_genes:
                    gene_symbols[position] = resolved_genes[sanitized_gene_id]

            output_lines[line_index] = '\t'.join(pathway_info + gene_symbols)

        final_unknown_genes = {sanitized_gene_id for sanitized_gene_id, symbol in
                               resolved_genes.items() if symbol == "Unknown"}
    else:
        final_unknown_genes = set()

    if final_unknown_genes:
        os.makedirs("./output/support", exist_ok=True)
        unknown_genes_file = os.path.join('./output/support/unknown_genes.txt')
        with open(unknown_genes_file, 'w') as unknown_file:
            unknown_file.write('\n'.join(final_unknown_genes))

        print(f"These genes are still unknown: {final_unknown_genes}")

    return output_lines, list(final_unknown_genes)


# Database Functions
def initialize_database(
        db_path: str = './database/reference_chunks.db'
) -> sqlite3.Connection:
    """
    Initializes the SQLite database and creates the 'chunks' table if it does not exist.

    Args:
        db_path: The path to the SQLite database file.

    Returns:
        A connection object to the SQLite database.
    """
    if not os.path.exists('./database'):
        os.makedirs('./database')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


def insert_chunk(
    conn: sqlite3.Connection,
    file_name: str,
    chunk_index: int,
    text: str
) -> int:
    """
    Inserts a text chunk into the database.

    Args:
        conn: The SQLite database connection.
        file_name: The name of the file from which the chunk was extracted.
        chunk_index: The index of the chunk within the file.
        text: The text content of the chunk.

    Returns:
        The row ID of the inserted chunk.
    """
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chunks (file_name, chunk_index, text)
        VALUES (?, ?, ?)
    ''', (file_name, chunk_index, text))
    conn.commit()
    return cursor.lastrowid


def fetch_chunks(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    """
    Retrieves all chunks from the database.

    Args:
        conn: The SQLite database connection.

    Returns:
        A list of tuples containing the chunk ID and text.
    """
    cursor = conn.cursor()
    cursor.execute('SELECT id, text FROM chunks')
    return cursor.fetchall()


def fetch_chunks_by_ids(
    conn: sqlite3.Connection,
    ids: List[int]
) -> Dict[int, str]:
    """
    Retrieves text chunks from the database that match the specified chunk IDs.

    Args:
        conn: The SQLite database connection.
        ids: A list of integer chunk IDs to fetch.

    Returns:
        A dictionary mapping each chunk ID (int) to its corresponding text content (str).
        If 'ids' is empty, returns an empty dictionary.
    """

    if not ids:
        return {}
    placeholder = ','.join('?' for _ in ids)
    query = f"SELECT id, text FROM chunks WHERE id IN ({placeholder})"
    cursor = conn.cursor()
    cursor.execute(query, ids)
    rows = cursor.fetchall()
    return {row[0]: row[1] for row in rows}


# FAISS Functions
def save_faiss_index(
    index: Any,  # faiss.IndexIDMap
    index_path: str = './database/faiss_index.bin'
) -> None:
    """
    Saves the FAISS index to a file.

    Args:
        index: The FAISS index to save.
        index_path: The file path where the FAISS index will be saved.
    """
    try:
        faiss.write_index(index, index_path)
    except RuntimeError as e:
        # Remove a possibly truncated file so the next run can recover cleanly.
        if os.path.exists(index_path) and os.path.getsize(index_path) == 0:
            os.remove(index_path)
        raise RuntimeError(f"Failed to write FAISS index to {index_path}: {e}")


def load_faiss_index(
    embedding_dim: int,
    index_path: str = './database/faiss_index.bin'
) -> Any:  # faiss.IndexIDMap
    """
    Loads a FAISS index from a file. If the file does not exist, creates a new IndexIDMap based on an IndexFlatIP.

    Args:
        embedding_dim: The dimension of the embeddings.
        index_path: The file path from which to load the FAISS index.

    Returns:
        A FAISS index with ID mapping.
    """
    if os.path.exists(index_path):
        # Zero-byte files can be left behind after interrupted writes.
        if os.path.getsize(index_path) == 0:
            os.remove(index_path)
            print(f"Deleted empty FAISS index at {index_path}; creating a new index.")
            faiss_index = faiss.IndexFlatIP(embedding_dim)
            return faiss.IndexIDMap(faiss_index)

        index = faiss.read_index(index_path)
        if not isinstance(index, faiss.IndexIDMap):
            raise ValueError(
                f"Loaded FAISS index is of type {type(index)}, but IndexIDMap is required.")
    else:
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIDMap(faiss_index)
        print("New FAISS IndexIDMap created.")
    return index


def initialize_faiss_index(
    embedding_dim: int,
    index_path: str
) -> Any:  # faiss.IndexIDMap
    """
    Load or recreate a FAISS index using the specified embedding dimension and index path.
    If the existing index is invalid, it is deleted and a new one is created.

    Args:
        embedding_dim: The dimension of the embeddings.
        index_path: The file path for the FAISS index.

    Returns:
        A valid FAISS index.
    """
    try:
        return load_faiss_index(embedding_dim, index_path=index_path)
    except (ValueError, RuntimeError):
        print(
            "Deleting existing FAISS index and creating a new one with IndexIDMap.")
        if os.path.exists(index_path):
            os.remove(index_path)
            print(f"Deleted FAISS index at {index_path}")
        return load_faiss_index(embedding_dim, index_path=index_path)


# Chunking Functions
def chunk_documents(documents: List[str]) -> List[str]:
    """
    Splits documents into chunks by breaking them into non-empty lines.

    Args:
        documents: A list of document strings.

    Returns:
        A list of text chunks derived from the input documents.
    """
    if not documents:
        print("No documents to chunk. Returning an empty list.")
        return []

    chunked_docs = []
    for idx, document in enumerate(documents):
        print(
            f"Processing Document {idx + 1} with length {len(document)} characters.")

        lines = document.split("\n")
        chunked_docs.extend(line for line in lines if
                            line.strip())

    print(
        f"Total number of chunks: {len(chunked_docs)} (should match line count)")
    return chunked_docs


def chunk_pdfs(
    single_document: List[Tuple[str, str, str]],
    gene_list: Optional[List[str]] = None,
    target_length: int = 1000
) -> List[str]:
    """
    Splits the text content of PDF documents into chunks based on sentence tokenization
    and a target minimum length. Optionally filters chunks by the presence of any gene from a provided list.

    Args:
        single_document: A list of tuples, where each tuple contains:
            - file_name: The name of the PDF file (str).
            - content: The extracted text content from that PDF (str).
            - file_hash: A hash of the PDF file contents (str).
        gene_list: An optional list of gene names (strings) to filter chunks.
                   If provided, only chunks containing at least one gene from this list are retained.
        target_length: The target minimum length (in characters) of each chunk (default: 1000).

    Returns:
        A list of text chunks (each chunk is a string) extracted from the PDF(s).
    """
    file_name, content, file_hash = single_document[0]
    sentences = sent_tokenize(content)
    if not sentences:
        return []

    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)

        if current_length >= target_length:
            chunk_text = " ".join(current_chunk)
            if not gene_list or any(gene in chunk_text for gene in gene_list):
                chunks.append(chunk_text)
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if not gene_list or any(gene in chunk_text for gene in gene_list):
            chunks.append(chunk_text)

    return chunks


# Embedding & Loading Functions
def load_file_log(
    log_path: str = './logs/file_log.json'
) -> Dict[str, Any]:
    """
    Loads a file log from a JSON file, creating the log directory and file if necessary.

    If the log file does not exist, initializes an empty dictionary and writes it to disk.

    Returns:
        A dictionary representing the file log (filename → metadata).
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory at '{log_dir}'.")
        except Exception as e:
            print(f"Failed to create log directory '{log_dir}': {e}")
            return {}

    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as file_logs:
                file_log = json.load(file_logs)
        except json.JSONDecodeError:
            file_log = {}
            print(f"File log at '{log_path}' is corrupted or empty. Initialized with an empty log.")
            save_file_log(file_log, log_path)
        except Exception as e:
            print(f"An error occurred while loading the file log: {e}")
            file_log = {}
    else:
        file_log = {}
        print(f"No existing file log found. Creating a new file log at '{log_path}'.")
        save_file_log(file_log, log_path)

    return file_log



def save_file_log(
    file_log: dict,
    log_path: str = './logs/file_log.json'
) -> None:
    """
    Saves the file log to a JSON file, ensuring the log directory exists.

    Args:
        file_log: The file log dictionary to save.
        log_path: The path to the JSON file where the log will be saved.
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory at '{log_dir}'.")
        except Exception as e:
            print(f"Failed to create log directory '{log_dir}': {e}")
            return

    try:
        with open(log_path, 'w', encoding='utf-8') as file_logs:
            json.dump(file_log, file_logs, indent=4)
    except Exception as e:
        print(f"An error occurred while saving the file log: {e}")


def load_embeddings_model_and_tokenizer(
    force_download: bool = False
) -> Tuple[Any, Any]:
    """
    Loads and returns a tokenizer and model from pretrained sources using the Transformers library.

    Args:
        force_download: If True, forces re-downloading of the model and tokenizer.

    Returns:
        A tuple containing the tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name, force_download=force_download)
    embeddings_model = AutoModel.from_pretrained(embeddings_model_name, force_download=force_download)
    return tokenizer, embeddings_model


# TODO See whether you can edit it for more modular approach
def load_gz_files(
    data_dir: str = './data/GSEA/external_gene_data'
) -> List[Tuple[str, str, str]]:
    """
    Loads documents from files in the specified directory that are either gzipped or in .gmt format.
    Computes a file hash for each document and returns a list of documents with their metadata.

    Args:
        data_dir: The directory containing the gene data files.

    Returns:
        A list of tuples, each containing the file name, document content, and file hash.
    """
    files = [
        gz_files for gz_files in os.listdir(data_dir)
        if
        gz_files.endswith('.gz') or (gz_files.startswith('converted') and gz_files.endswith('.gmt'))
    ]

    documents = []
    if len(files) > 0:
        print(f"Found {len(files)} files in '{data_dir}'.")

    for file in files:
        file_path = os.path.join(data_dir, file)
        file_hash = compute_file_hash(file_path)

        if file.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as gz_files:
                content = gz_files.read().strip()
        else:
            with open(file_path, 'r', encoding='utf-8') as gz_files:
                content = gz_files.read().strip()

        if content:
            lines = content.splitlines()
            document = "\n".join(lines)
            documents.append((file, document, file_hash))
            print(
                f"Loaded '{file}' with {len(lines)} lines. Document length: {len(document)} characters.")
        else:
            print(f"Skipped empty file '{file}'.")

    return documents


def load_pdf_files(
        pdf_dir: str = './data/PDF',
        file_log: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, str, str]]:
    """
    Loads PDF documents from the specified directory, skipping those already recorded in the file log.

    Args:
        pdf_dir: The directory containing PDF files (default: "./data/PDF").
        file_log: An optional dictionary mapping PDF filenames (str) to metadata. PDFs present in this log
                  will be skipped to avoid reprocessing.

    Returns:
        A list of tuples, each containing:
            - file_name: The PDF file name (str).
            - content: The full extracted text content of the PDF (str).
            - file_hash: A SHA-256 hash of the PDF file (str).
    """
    pdf_documents = []
    if not os.path.exists(pdf_dir):
        print(f"No PDF directory found at '{pdf_dir}'. Skipping PDF import.")
        return pdf_documents

    file_log = file_log or {}

    # Debug: Show what's in the file log for PDFs
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"\nDEBUG: Found {len(pdf_files)} PDF files in {pdf_dir}")

    skipped_count = 0
    processed_count = 0

    for file in pdf_files:
        if file in file_log:
            print(
                f"  - Skipping '{file}' (already in file log with hash: {file_log[file].get('hash', 'unknown')[:8]}...)")
            skipped_count += 1
            continue

        file_path = os.path.join(pdf_dir, file)
        try:
            print(f"  - Processing '{file}'...")
            reader = PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            content = "\n".join(pages_text).strip()

            if content:
                file_hash = compute_file_hash(file_path)
                pdf_documents.append((file, content, file_hash))
                print(
                    f"    ✓ Loaded PDF '{file}' with {len(pages_text)} pages. Document length: {len(content)} characters.")
                processed_count += 1
            else:
                print(f"    ✗ Skipped empty PDF '{file}'.")
        except Exception as e:
            print(f"    ✗ Failed to read PDF '{file}': {e}")

    print(f"PDF Summary: {processed_count} new PDFs loaded, {skipped_count} skipped (already processed)")

    return pdf_documents


def embed_documents(
        conn: sqlite3.Connection,
        index: Any,
        tokenizer: Any,
        embeddings_model: Any,
        data_dir: str,
        batch_size: int,
        log_path: str = './logs/file_log.json',
        pdf_dir: str = './data/PDF'
) -> None:
    """
    Embeds text from documents (both gzipped and PDF files) using a transformer model,
    updates the FAISS index with the embeddings, and records file metadata in a log.

    Args:
        conn: The SQLite database connection.
        index: The FAISS index to update.
        tokenizer: The tokenizer for preparing text inputs.
        embeddings_model: The transformer model for generating embeddings.
        data_dir: Directory containing gzipped documents.
        batch_size: The batch size for embedding computation.
        log_path: The path to the file log.
        pdf_dir: Directory containing PDF files.
    """
    file_log = load_file_log(log_path=log_path)

    files_documents = load_gz_files(data_dir=data_dir)
    pdf_documents = load_pdf_files(pdf_dir=pdf_dir, file_log=file_log)

    # Debug: Check what we loaded
    print(f"DEBUG: Loaded {len(files_documents)} .gz/.gmt files from '{data_dir}'")
    if files_documents:
        print(f"  Files: {[doc[0] for doc in files_documents[:3]]}...")  # Show first 3 filenames
    print(f"DEBUG: Loaded {len(pdf_documents)} new PDF files from '{pdf_dir}'")
    if pdf_documents:
        print(f"  PDFs: {[doc[0] for doc in pdf_documents[:3]]}...")  # Show first 3 filenames

    all_documents = files_documents + pdf_documents

    if not all_documents:
        print("WARNING: No documents found to process!")
        print(f"  Checked data_dir: {data_dir}")
        print(f"  Checked pdf_dir: {pdf_dir}")
        return

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    embeddings_model.to(device)
    embeddings_model.eval()

    total_chunks_inserted = 0

    for file, document, file_hash in all_documents:
        if file in file_log and file_log[file]['hash'] == file_hash:
            print(f"Skipping already processed file: {file}")
            continue
        else:
            print(f"Processing file: {file}")

        lines = document.splitlines()
        if not lines:
            print(f"No lines found in file '{file}'. Skipping embedding.")
            continue

        if file.lower().endswith('.pdf'):
            chunks = chunk_pdfs([(file, document, file_hash)], gene_list=None)
        else:
            document_content = "\n".join(lines[1:]) if file.endswith('txt.gz') else "\n".join(lines)
            chunks = chunk_documents([document_content])

        if not chunks:
            print(f"No chunks created for file '{file}'. Skipping embedding.")
            continue

        print(f"Created {len(chunks)} chunks from '{file}'")
        print(f"Embedding chunks of '{file}' on device: {device}")
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(chunks), batch_size),
                          desc=f"Processing {file} in batches"):
                batch = chunks[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                                   padding=True, max_length=512).to(device)
                outputs = embeddings_model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()).float()
                sum_embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                embeddings.extend(batch_embeddings)

        if not embeddings:
            print(f"No embeddings generated for file '{file}'.")
            continue

        embeddings_np = np.vstack(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)

        chunk_ids = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = insert_chunk(conn, file, idx, chunk_text)
            chunk_ids.append(chunk_id)
            total_chunks_inserted += 1

        print(f"Inserted {len(chunk_ids)} chunks into database for file '{file}'")

        faiss_ids = np.array(chunk_ids).astype('int64')
        index.add_with_ids(embeddings_np, faiss_ids)

        file_log[file] = {
            'hash': file_hash,
            'num_embeddings': embeddings_np.shape[0]
        }

    print(f"DEBUG: Total chunks inserted into database: {total_chunks_inserted}")

    save_faiss_index(index, index_path='./database/faiss_index.bin')
    save_file_log(file_log, log_path=log_path)


# BM25 and Query Functions
def tokenize(text: str) -> List[str]:
    """
    Tokenizes the input text into lowercase words, excluding common stopwords.

    Args:
        text: The text string to tokenize.

    Returns:
        A list of tokens.
    """
    return [word for word in re.findall(r'\b[\w-]+\b', text.lower()) if word not in stop_words]


def build_bm25_index(
    conn: sqlite3.Connection
) -> Tuple[BM25Okapi, List[int], List[str]]:
    """
    Builds a BM25 index for the text chunks stored in the database.
    """
    cursor = conn.cursor()
    cursor.execute('SELECT id, text FROM chunks')
    data = cursor.fetchall()

    # Debug: Check if we have any chunks
    if not data:
        print("WARNING: No chunks found in database!")
        print("Creating empty BM25 index...")
        # Return empty structures to avoid crash
        return BM25Okapi([["dummy"]]), [], []

    chunk_ids = []
    chunk_texts = []
    for row in data:
        chunk_ids.append(row[0])
        chunk_texts.append(row[1])

    print(f"Building BM25 index with {len(chunk_texts)} chunks")
    tokenized_corpus = [tokenize(doc) for doc in chunk_texts]

    # Filter out empty documents
    non_empty_corpus = []
    non_empty_ids = []
    non_empty_texts = []
    for tokens, chunk_id, chunk_text in zip(tokenized_corpus, chunk_ids, chunk_texts):
        if tokens:  # Only include non-empty token lists
            non_empty_corpus.append(tokens)
            non_empty_ids.append(chunk_id)
            non_empty_texts.append(chunk_text)

    if not non_empty_corpus:
        print("WARNING: All documents are empty after tokenization!")
        return BM25Okapi([["dummy"]]), [], []

    bm25 = BM25Okapi(non_empty_corpus)
    return bm25, non_empty_ids, non_empty_texts


def query_bm25_index(
    query: str,
    index: BM25Okapi,
    chunk_ids: List[int],
    top_k: int = 1000
) -> Tuple[List[int], List[float], Dict[int, Any]]:
    """
        Queries the BM25 index using the provided query text and returns the top scoring documents along with details.

        Args:
            query: The text query.
            index: The BM25 index object.
            chunk_ids: List of chunk IDs corresponding to the BM25 index.
            top_k: The number of top results to return.

        Returns:
            A tuple containing:
                - A list of top chunk IDs.
                - A list of their corresponding scores.
                - A dictionary with detailed token contributions for each chunk.
        """
    tokens = tokenize(query)
    scores = index.get_scores(tokens)
    order = np.argsort(scores)[::-1][:top_k]

    top_ids = [chunk_ids[i] for i in order]
    top_scores = [float(scores[i]) for i in order]

    details: Dict[int, Any] = {}
    for i, s in zip(order, top_scores):
        cid = chunk_ids[i]
        tok_details = []
        for t in tokens:
            tf = index.doc_freqs[i].get(t, 0)
            if tf:
                idf, b, avgdl, k1 = index.idf[t], index.b, index.avgdl, index.k1
                numerator = (k1+1)*tf
                denominator = k1*(1-b + b*(index.doc_len[i]/avgdl)) + tf
                tok_details.append(f"{t} score={idf*(numerator/denominator):.4f}")
        details[cid] = {"score": round(s, 4), "tokens": tok_details or None}

    return top_ids, top_scores, details


def query_faiss_index(
    query_text: str,
    index: Any,  # faiss.IndexIDMap
    tokenizer: Any,
    embeddings_model: Any,
    top_k: int,
    force_download: bool = False
) -> Tuple[List[int], List[float]]:
    """
    Computes the embedding for a query and searches the FAISS index to retrieve the closest document chunks.

    Args:
        query_text: The query string.
        index: The FAISS index.
        tokenizer: The tokenizer for preparing the query.
        embeddings_model: The transformer model for generating the query embedding.
        top_k: The number of top results to retrieve.
        force_download: If True, forces re-downloading of model/tokenizer if needed.

    Returns:
        A tuple containing the list of top document IDs and their corresponding distances.
    """
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    loaded_model = AutoModel.from_pretrained(embeddings_model_name, force_download=force_download)
    loaded_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name, force_download=force_download)
    if embeddings_model.config != loaded_model.config:
        print("The models have different configurations.")
        embeddings_model = AutoModel.from_pretrained(embeddings_model_name, force_download=force_download)
    if str(tokenizer) != str(loaded_tokenizer):
        print("The tokenizers have different configurations.")
        tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name, force_download=force_download)
    embeddings_model.to(device)
    embeddings_model.eval()
    with torch.no_grad():
        inputs = tokenizer([query_text], return_tensors="pt", truncation=True,
                           padding=True, max_length=512).to(device)
        outputs = embeddings_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    top_ids = [int(id_) for id_ in indices[0] if id_ != -1]
    top_distances = [float(dist) for dist in distances[0] if dist != -1]
    return top_ids, top_distances


# Document Ranking and Combination
def create_top_faiss_docs(
    expanded_queries: List[str],
    index: Any,
    tokenizer: Any,
    embeddings_model: Any,
    top_k: int
) -> List[Tuple[int, Tuple[float, str]]]:
    """
    Retrieves top documents from the FAISS index for each expanded query and computes average distances.

    Args:
        expanded_queries: A list of expanded query strings.
        index: The FAISS index.
        tokenizer: The tokenizer for processing the queries.
        embeddings_model: The transformer model for generating embeddings.
        top_k: The number of top documents to retrieve per query.

    Returns:
        A list of tuples containing document IDs and their average distances.
    """
    doc_distances = {}

    for eq in expanded_queries:
        top_ids, distances = query_faiss_index(eq, index, tokenizer, embeddings_model, top_k=top_k)
        for doc_id, distance in zip(top_ids, distances):
            if doc_id not in doc_distances:
                doc_distances[doc_id] = []
            doc_distances[doc_id].append(distance)

    faiss_doc_distance_dict = {}
    for doc_id, dist_list in doc_distances.items():
        avg_dist = sum(dist_list) / len(dist_list)
        faiss_doc_distance_dict[doc_id] = (avg_dist, "avg")

    sorted_faiss_docs = sorted(faiss_doc_distance_dict.items(), key=lambda item: item[1][0])
    top_faiss_docs = sorted_faiss_docs[:top_k]

    return top_faiss_docs


def create_top_bm25_docs(
    expanded_queries: List[str],
    bm25_index: BM25Okapi,
    chunk_ids: List[int],
    top_k: int
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Retrieves top documents from the BM25 index for each expanded query and aggregates their scores.

    Args:
        expanded_queries: A list of expanded query strings.
        bm25_index: The BM25 index object.
        chunk_ids: List of chunk IDs corresponding to the BM25 index.
        top_k: The number of top documents to retrieve per query.

    Returns:
        A list of tuples where each tuple contains a document ID and its aggregated BM25 score and details.
    """
    bm25_doc_scores = {}

    for eq in expanded_queries:
        top_ids, top_scores, detailed_scores = query_bm25_index(eq, bm25_index, chunk_ids, top_k=top_k)
        for doc_id, details in detailed_scores.items():
            if doc_id not in bm25_doc_scores:
                bm25_doc_scores[doc_id] = {
                    'scores': [],
                    'tokens': "",
                    'best_tokens_score': float('-inf'),
                    'source_queries': []
                }

            bm25_doc_scores[doc_id]['scores'].append(details['score'])

            bm25_doc_scores[doc_id]['source_queries'].append(eq)

            if details['score'] > bm25_doc_scores[doc_id]['best_tokens_score']:
                bm25_doc_scores[doc_id]['best_tokens_score'] = details['score']
                bm25_doc_scores[doc_id]['tokens'] = (
                    ', '.join(details['tokens']) if isinstance(details['tokens'], list) else details['tokens']
                )

    for doc_id, info in bm25_doc_scores.items():
        avg_score = sum(info['scores']) / len(info['scores'])
        info['score'] = avg_score

    sorted_bm25_docs = sorted(bm25_doc_scores.items(),
                              key=lambda item: item[1]['score'],
                              reverse=True)
    top_bm25_docs = sorted_bm25_docs[:top_k]

    return top_bm25_docs


def weighted_rrf(
    top_bm25_docs: List[Tuple[int, Dict[str, Any]]],
    top_faiss_docs: List[Tuple[int, Tuple[float, str]]],
    weight_faiss: float,
    weight_bm25: float
) -> Dict[int, Dict[str, Any]]:
    """
    Combine BM25 and FAISS scores using specified weights.

    Args:
        top_bm25_docs: List of top documents from BM25.
        top_faiss_docs: List of top documents from FAISS.
        weight_faiss: Weight for FAISS scores.
        weight_bm25: Weight for BM25 scores.

    Returns:
        A dictionary mapping document IDs to their combined weighted scores.
    """
    total_weight = weight_faiss + weight_bm25
    weight_faiss /= total_weight
    weight_bm25 /= total_weight
    combined_scores = {}

    for bm25_doc in top_bm25_docs:
        doc_id, details = bm25_doc
        score = details["score"] * weight_bm25
        if doc_id in combined_scores:
            combined_scores[doc_id]["score"] += score
        else:
            combined_scores[doc_id] = {"score": score, "source_queries": details.get("source_queries", [])}

    for faiss_doc in top_faiss_docs:
        doc_id, (distance, _) = faiss_doc
        score = (1 / (1 + distance)) * weight_faiss
        if doc_id in combined_scores:
            combined_scores[doc_id]["score"] += score
        else:
            combined_scores[doc_id] = {"score": score, "source_queries": []}

    return combined_scores


def rank_and_retrieve_documents(
    rrf_scores: Dict[int, Dict[str, Any]],
    conn: sqlite3.Connection,
    top_faiss_docs: List[Tuple[int, Tuple[float, str]]],
    top_bm25_docs: List[Tuple[int, Dict[str, Any]]],
    amount_docs: int
) -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
    """
    Ranks documents based on combined RRF scores, retrieves the corresponding text chunks from the database,
    and returns them in ranked order.

    Args:
        rrf_scores: A dictionary of combined scores for documents.
        conn: The SQLite database connection.
        top_faiss_docs: List of top documents from FAISS.
        top_bm25_docs: List of top documents from BM25.
        amount_docs: The number of documents to retrieve.

    Returns:
        A tuple containing:
            - A list of ordered document chunks.
            - A dictionary with combined document details and their rankings.
    """
    sorted_rrf_scores = sorted(rrf_scores.items(),
                               key=lambda item: item[1]['score'],
                               reverse=True)[:amount_docs]

    unique_ids = {doc_id for doc_id, _ in sorted_rrf_scores}

    combined_docs = {}
    for rank, (doc_id, data) in enumerate(sorted_rrf_scores, start=1):
        score = data['score']
        source_queries = data.get('source_queries', [])
        combined_docs[doc_id] = {
            'retriever': [],
            'rank': rank,
            'score': score,
            'source_queries': source_queries
        }
        if any(faiss_doc[0] == doc_id for faiss_doc in top_faiss_docs):
            combined_docs[doc_id]['retriever'].append('FAISS')
        if any(bm25_doc[0] == doc_id for bm25_doc in top_bm25_docs):
            combined_docs[doc_id]['retriever'].append('BM25')

    # build the ID→text map in one go
    doc_id_to_chunk = fetch_chunks_by_ids(conn, list(unique_ids))

    # then simply iterate in rank order
    ordered_chunks = [
        doc_id_to_chunk[doc_id]
        for doc_id, _ in sorted_rrf_scores
        if doc_id in doc_id_to_chunk
    ]

    return ordered_chunks, combined_docs


# Query Expansion and Response Generation
def query_expansion(
    query_text: str,
    number: int
) -> List[str]:
    """
    Expands a given query into a specified number of alternative queries by generating synonyms and related phrases.

    Args:
        query_text: The original user query.
        number: The number of expanded queries to generate.

    Returns:
        A list of expanded query strings.
    """
    system_instruction = f"""You are an expert in query expansion for biomedical literature searches. Given a user's 
query, generate exactly {number} alternative queries that capture related concepts, synonyms, and relevant expansions. 
Each alternative query should be concise and directly related to the original query.

Example of a query expansion:
User query: "GGT5 in glutathione metabolism"

Possible expanded queries:
1. "Gamma-glutamyltransferase 5 role in glutathione metabolism"
2. "GGT5 enzyme function in glutathione pathway"
3. "GGT5 and its involvement in glutathione homeostasis"
4. "Impact of GGT5 on glutathione biosynthesis and metabolism"

Consider the context of the user query and ensure that the expanded queries are relevant to the topic.
Only return the expanded queries, no explanation is needed.
    """
    expanded_queries = []
    if number > 0:
        prompt = (
            f"Based on the user's query, provide exactly {number} expanded queries that include related terms,"
            f" synonyms, and relevant expansions.\n\nUser query: \"{query_text}\"")

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        save = False
        answer = query_llm(messages, prompt, system_instruction, save, generation_model=generation_model,
                           query_range=1)
        if not answer:
            return []
        expanded_queries = answer.strip().split('\n')
        expanded_queries = [q.lstrip("-•*").lstrip("0123456789. ").strip() for q in expanded_queries if q.strip()]
        if len(expanded_queries) > number:
            expanded_queries = expanded_queries[:number]
        elif len(expanded_queries) < number:
            print(f"Warning: Expected {number} queries but received {len(expanded_queries)}.")

    return expanded_queries


def build_search_parameters(
    mode: str = "on",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_results: int = 20,
    academic: bool = False,
    academic_rss_links: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Constructs a dictionary of search parameters for a generic web/news search.

    Args:
        mode: Search mode ("on" or "off"), determining whether to perform live searches (default: "on").
        from_date: Optional start date filter in "YYYY-MM-DD" format.
        to_date: Optional end date filter in "YYYY-MM-DD" format.
        max_results: Maximum number of search results to return (default: 20).
        academic: If True, include academic RSS sources in the search (default: False).
        academic_rss_links: An optional list of RSS feed URLs (strings) for academic sources.
                            If None and academic=True, a default ArXiv feed will be used.

    Returns:
        A dictionary of parameters suitable for passing to a search API, for example:
            {
                "mode": ...,
                "return_citations": True,
                "max_search_results": ...,
                "sources": [...],
                "from_date": ...,
                "to_date": ...
            }
    """
    sources = [
        {"type": "web"},
        {"type": "x"},
        {"type": "news"},
    ]
    if academic:
        default_arxiv = "https://export.arxiv.org/rss/cs"
        rss_links = academic_rss_links or [default_arxiv]
        for link in rss_links:
            sources.append({"type": "rss", "links": [link]})

    params: dict = {
        "mode": mode,
        "return_citations": True,
        "max_search_results": max_results,
        "sources": sources,
    }
    if from_date:
        params["from_date"] = from_date
    if to_date:
        params["to_date"] = to_date

    return params


def get_generation_model_dir(model: str) -> str:
    """
    Maps a model identifier string to the corresponding subdirectory name used for storing outputs.

    Args:
        model: The generation model name (e.g., "o4-mini", "gpt-4", "grok-3-mini-beta").

    Returns:
        A string representing the directory name under "./output/test_files/" where
        this model's outputs should be saved.
    """
    if model.startswith("grok"):
        return "Grok 3-mini-beta"
    if model == "gpt-4o-mini-search-preview" or model.startswith("gpt-4"):
        return "OpenAI websearch"
    if model.startswith("o"):
        if "o4-mini" in model:
            return "OpenAI o4-mini"
        if "o3-mini" in model:
            return "OpenAI o3-mini"
        if "o4" in model:
            return "OpenAI o4"
        return "OpenAI o3"
    if model.startswith("deepseek"):
        return "DeepSeek R1"
    if model.startswith("claude"):
        return "Claude"
    if model.lower().startswith("gemini"):
        return "Gemini"
    return "OpenAI"


def should_set_default_temperature(model: str) -> bool:
    """
    Determine whether an OpenAI-compatible model should receive an explicit default temperature.

    GPT-5 family chat models reject explicit non-default temperature values in this code path,
    so they must be called without a temperature parameter unless one is intentionally supplied
    elsewhere with a supported value.
    """
    normalized_model = model.lower()
    return not normalized_model.startswith("gpt-5")


def query_llm(
    messages: List[Dict[str, Any]],
    system_instruction_for_response: str,
    prompt: str,
    save: bool,
    generation_model: str,
    query_range: int,
    gene_count: Optional[int] = None,
    **kwargs: Any
) -> Optional[str]:
    """
    Unified LLM query function that supports Grok, OpenAI, DeepSeek, Claude, and Gemini.
    Retries up to 'query_range' times, measures each call's duration, and optionally saves
    the model output to disk.

    Args:
        messages: A list of message dictionaries in the form {'role': str, 'content': str}
                  to send to the chat completion endpoint.
        system_instruction_for_response: The system‐level instruction guiding LLM behavior.
        prompt: The user prompt to append after the system instruction (str).
        save: If True, save each LLM response to a file under "./output/test_files/<model_dir>/".
        generation_model: The name of the model to query (e.g., "o4-mini").
        query_range: Number of retry attempts for the LLM call.
        gene_count: Optional integer indicating how many genes were used; incorporated into saved filename.
        **kwargs: Additional model-specific keyword arguments (e.g., temperature, max_tokens).

    Returns:
        The final LLM-generated string (str) if successful, or None if all attempts fail.

    Raises:
        Any exceptions from the underlying HTTP/SDK calls are caught and printed; no exception is propagated.
    """

    answers: list[str] = []
    for i in range(1, query_range + 1):
        try:
            start_time = time.perf_counter()
            model_dir = get_generation_model_dir(generation_model)
            # ── Grok via raw HTTP ───────────────────────────────────────────────
            if model_dir == "Grok 3-mini-beta":
                api_key = os.getenv("XAI_API_KEY")
                endpoint = "https://api.x.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                # Grok-specific search params
                academic = kwargs.pop('academic', False)
                academic_rss = kwargs.pop('academic_rss_links', None)
                from_date = kwargs.pop('from_date', None)
                to_date = kwargs.pop('to_date', None)
                max_results = kwargs.pop('max_results', 20)
                mode = kwargs.pop('mode', 'on')
                temperature_grok = kwargs.pop('temperature', 0)

                payload = {
                    "model": generation_model,
                    "messages": messages,
                    "search_parameters": build_search_parameters(
                        mode=mode,
                        from_date=from_date,
                        to_date=to_date,
                        max_results=max_results,
                        academic=academic,
                        academic_rss_links=academic_rss
                    ),
                    "temperature": temperature_grok
                }
                resp = requests.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]

            elif model_dir == "OpenAI":
                openai_params: Dict[str, Any] = {
                    "model": generation_model,
                    "messages": messages,
                    **kwargs
                }
                if (
                    "temperature" not in openai_params
                    and should_set_default_temperature(generation_model)
                ):
                    openai_params["temperature"] = 0

                response = client_open_ai.chat.completions.create(  # type: ignore
                    **openai_params
                )
                answer = response.choices[0].message.content

            elif model_dir.startswith("OpenAI o"):
                adjusted = system_instruction_for_response + prompt
                response = client_open_ai.chat.completions.create(  # type: ignore
                    model=generation_model,
                    messages=[{"role": "user", "content": adjusted}],
                    max_completion_tokens=32768,
                    reasoning_effort="low",
                    **kwargs
                )
                answer = response.choices[0].message.content

            elif model_dir == "DeepSeek R1":
                response = client_deepseek.chat.completions.create(  # type: ignore
                    model=generation_model,
                    messages=messages,
                    stream=False,
                    temperature=0,
                    **kwargs
                )
                answer = response.choices[0].message.content

            # OpenAI SDK is in beta at the moment, maybe in the future, more arguments will be included
            elif model_dir == "Claude":
                response = client_claude.chat.completions.create(  # type: ignore
                    model=generation_model,
                    messages=messages,  # type: ignore
                )
                answer = response.choices[0].message.content

            elif model_dir == "Gemini":
                response = client_gemini.chat.completions.create(  # type: ignore
                    model=generation_model,
                    messages=messages,  # type: ignore
                    temperature=0.0
                )
                answer = response.choices[0].message.content
            else:
                # Fallback to OpenAI
                fallback_params: Dict[str, Any] = {
                    "model": generation_model,
                    "messages": messages,
                    **kwargs
                }
                if (
                    "temperature" not in fallback_params
                    and should_set_default_temperature(generation_model)
                ):
                    fallback_params["temperature"] = 0

                response = client_open_ai.chat.completions.create(  # type: ignore
                    **fallback_params
                )
                answer = response.choices[0].message.content

            duration = round(time.perf_counter() - start_time, 2)

        except Exception as e:
            print(f"[{generation_model}][iter {i}] Error: {e}")
            continue

        # Save to disk if requested
        if save:
            out_dir = os.path.join("./output/test_files", model_dir)
            os.makedirs(out_dir, exist_ok=True)
            if gene_count is not None:
                file_name = f"{generation_model}-GSEA-{gene_count}-{i}-{duration}.txt"
            else:
                file_name = f"{generation_model}-{i}-{duration}.txt"

            path = os.path.join(out_dir, file_name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(answer)

        answers.append(answer)

    return answers[-1] if answers else None


def generate_llm_response(
    query_text: str,
    gene_list_string: str,
    conn: sqlite3.Connection,
    index: Any,
    tokenizer: Any,
    embeddings_model: Any,
    bm25_index: Any,
    bm25_chunk_ids: List[int],
    weight_faiss: float,
    weight_bm25: float,
    system_instruction_for_response: str,
    gene_count: Optional[int] = None
) -> Tuple[
    Optional[str],
    List[str],
    Dict[int, Dict[str, Any]],
    Dict[int, Any],
    Dict[int, Any]
]:
    """
    Generates a response from a language model by performing the following steps:
      1. Expands the input query into multiple variants.
      2. Retrieves relevant document chunks using FAISS and BM25.
      3. Combines scores using weighted Reciprocal Rank Fusion (RRF).
      4. Builds a prompt containing the retrieved documents plus the original query.
      5. Queries the specified LLM API (OpenAI, Claude, Gemini, etc.) to generate a final answer.

    Args:
        query_text: The original user query (str).
        gene_list_string: A comma‐separated string of gene names (str).
        conn: An active SQLite database connection for chunk storage.
        index: The FAISS index object for embedding-based retrieval.
        tokenizer: The tokenizer used to preprocess text for embeddings.
        embeddings_model: The transformer model used to generate embeddings.
        bm25_index: The BM25 index object for lexical retrieval.
        bm25_chunk_ids: A list of integer chunk IDs corresponding to the BM25 index.
        weight_faiss: Weight (float) assigned to FAISS-based scores in RRF.
        weight_bm25: Weight (float) assigned to BM25-based scores in RRF.
        system_instruction_for_response: The system‐level instruction string to guide LLM generation.
        gene_count: Optional integer representing how many genes were processed; used in saved filenames.

    Returns:
        A tuple containing:
            - answer: The final generated answer from the LLM (or None on failure).
            - document_references: A list of strings, each summarizing a retrieved chunk and its source.
            - rrf_scores: A dictionary mapping chunk IDs (int) to their combined RRF score and metadata.
            - bm25_scores: A dictionary mapping chunk IDs (int) to their BM25-specific score details.
            - faiss_scores: A dictionary mapping chunk IDs (int) to their FAISS distance values.
    """
    # Expand the retrieval query
    expanded_queries = query_expansion(query_text, number=number_of_expansions)
    if not expanded_queries:
        expanded_queries = [query_text]
    else:
        expanded_queries.append(query_text)
    query_expanded_queries = [f"{eq} {gene_list_string}" for eq in expanded_queries]
    query_text = query_text + gene_list_string

    # Retrieve documents using FAISS and BM25
    top_faiss_docs = create_top_faiss_docs(query_expanded_queries, index, tokenizer, embeddings_model, top_k=50)
    top_bm25_docs = create_top_bm25_docs(query_expanded_queries, bm25_index, bm25_chunk_ids, top_k=50)

    # Combine scores using weighted RRF
    rrf_scores = weighted_rrf(top_bm25_docs, top_faiss_docs, weight_faiss, weight_bm25)
    retrieved_chunks_ordered, combined_docs = rank_and_retrieve_documents(
        rrf_scores, conn, top_faiss_docs, top_bm25_docs, amount_docs=amount_docs
    )

    if len(retrieved_chunks_ordered) < amount_docs:
        print(f"Warning: Retrieved only {len(retrieved_chunks_ordered)} documents but expected {amount_docs}.")

    document_references = []
    for doc_id in combined_docs.keys():
        chunk_text = retrieved_chunks_ordered.pop(0) if retrieved_chunks_ordered else "No text available"
        document_references.append(
            f"|Reference {combined_docs[doc_id]['rank']} ({' and '.join(combined_docs[doc_id]['retriever'])}) with "
            f"RRF Score: {combined_docs[doc_id]['score']:.4f}\n{chunk_text}"
        )
    combined_documents = "\n\n".join(document_references)

    prompt = (
        f"Based on the following documents, answer the question using both your knowledge and the provided "
        f"documents: {query_text}\n\nDocuments:\n{combined_documents}\n"
    )

    messages = [
        {"role": "system", "content": system_instruction_for_response},
        {"role": "user", "content": prompt}
    ]
    save_message = f"(role: system, content: {system_instruction_for_response}\nrole: user, content: {prompt})"
    os.makedirs("./output/support", exist_ok=True)
    with open("./output/support/messages.txt", "w", encoding="utf-8") as file:
        file.write(save_message)

    answer = query_llm(messages, system_instruction_for_response, prompt, save=True,
                       generation_model=generation_model, query_range=query_range, gene_count=gene_count)

    bm25_scores = dict([(doc_id, details['score']) for doc_id, details in top_bm25_docs])
    faiss_scores = {doc_id: distance for doc_id, (distance, eq) in top_faiss_docs}

    return answer, document_references, rrf_scores, bm25_scores, faiss_scores


def generate_response_and_save(
    query: str,
    gene_list_string: str,
    conn: sqlite3.Connection,
    index: Any,
    tokenizer: Any,
    embeddings_model: Any,
    bm25_index: Any,
    bm25_chunk_ids: List[int],
    weight_faiss: float,
    weight_bm25: float,
    system_instruction_for_response: str,
    gene_count: Optional[int] = None
) -> None:
    """
    Orchestrates a full retrieval‐augmented generation (RAG) workflow:
      1. Calls generate_llm_response(...) to retrieve documents and generate an LLM answer.
      2. Saves the LLM answer and retrieved document references to disk.
      3. Exports combined retrieval scores (RRF, BM25, FAISS) to an Excel file.
      4. Closes the database connection.

    Args:
        query: The original user query (str).
        gene_list_string: A comma‐separated string of gene names (str).
        conn: An active SQLite database connection.
        index: The FAISS index object.
        tokenizer: The tokenizer used for embeddings.
        embeddings_model: The transformer model used to embed text.
        bm25_index: The BM25 index object.
        bm25_chunk_ids: A list of integer chunk IDs for BM25 retrieval.
        weight_faiss: Weight (float) for FAISS scores in RRF combination.
        weight_bm25: Weight (float) for BM25 scores in RRF combination.
        system_instruction_for_response: The system‐level instruction string for the LLM.
        gene_count: Optional integer indicating gene count; included in filenames for saved outputs.

    Returns:
        None. Writes the LLM answer, document references, and scores to disk, then closes 'conn'.
    """
    answer, document_references, rrf_scores, bm25_scores, faiss_scores = generate_llm_response(
        query,  gene_list_string,
        conn, index, tokenizer, embeddings_model,
        bm25_index, bm25_chunk_ids,
        weight_faiss, weight_bm25,
        system_instruction_for_response, gene_count
    )

    if answer and answer != "Processing complete.":
        save_answer_to_file(answer, document_references)
        support_dir = Path("./output/support")
        support_dir.mkdir(parents=True, exist_ok=True)
        output_file = support_dir / "scores.xlsx"
        export_scores_to_excel(
            rrf_scores,
            bm25_scores,
            faiss_scores,
            file_name=str(output_file)
        )
    else:
        print("Failed to generate a response from the LLM.")
    conn.close()


# File Saving and Processing Helpers
def save_answer_to_file(
    answer: str,
    document_references: List[str],
    file_name: str = "./output/results/answer.txt"
) -> None:

    """
    Saves the generated answer and associated document references to text files.

    Args:
        answer: The generated answer string.
        document_references: A list of document reference strings.
        file_name: The file path where the answer will be saved.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding='utf-8') as answer_file:
        answer_file.write(answer)
    os.makedirs("./output/support", exist_ok=True)
    with open("./output/support/documents.txt", "w", encoding='utf-8') as answer_file:
        for idx, doc in enumerate(document_references, start=1):
            answer_file.write(f"{doc} \n\n")


def process_files_in_directory(
    data_dir: str,
    ncbi_json_dir: str
) -> None:
    """
    Processes all .gmt.gz files in the given directory by converting gene IDs to symbols and logging unknown genes.

    Args:
        data_dir: The directory containing the files to process.
        ncbi_json_dir: The directory to store JSON cache files for gene IDs.
    """
    """Process all .gmt.gz files in the given directory."""
    for file in os.listdir(data_dir):
        full_file_path = os.path.join(data_dir, file)
        if os.path.isfile(full_file_path) and file.endswith('.gmt.gz'):
            print(f"Processing file: {file}")

            converted_lines, unknown_genes = convert_gene_id_to_symbols(file,
                                                                        data_dir, ncbi_json_dir)
            print(
                f"Unknown genes saved to 'unknown_genes.txt'. Total unknown genes: {len(unknown_genes)}")


def embed_documents_and_save(
    index: Any,
    conn: sqlite3.Connection,
    tokenizer: Any,
    embeddings_model: Any,
    data_dir: str,
    batch_size: int,
    log_path: str,
    index_path: str
) -> None:
    """
    Embeds documents from the specified directory, updates the FAISS index with the embeddings,
    and then saves the updated index and closes the database connection.

    Args:
        index: The FAISS index.
        conn: The SQLite database connection.
        tokenizer: The tokenizer for processing text.
        embeddings_model: The transformer model for embedding computation.
        data_dir: The directory containing the documents.
        batch_size: The batch size for processing documents.
        log_path: The file path for the file log.
        index_path: The file path to save the FAISS index.
    """
    embed_documents(conn, index, tokenizer, embeddings_model, data_dir=data_dir,
                    batch_size=batch_size, log_path=log_path)
    save_faiss_index(index, index_path=index_path)
    conn.close()


def export_scores_to_excel(
    rrf_scores: Dict[int, Any],
    bm25_scores: Dict[int, Any],
    faiss_scores: Dict[int, Any],
    file_name: str = "./output/scores.xlsx"
) -> None:
    """
    Exports combined RRF, BM25, and FAISS scores to an Excel file.

    Args:
        rrf_scores: A dictionary of RRF scores.
        bm25_scores: A dictionary of BM25 scores.
        faiss_scores: A dictionary of FAISS scores.
        file_name: The file path where the Excel file will be saved.
    """
    all_doc_ids = set(rrf_scores.keys()).union(bm25_scores.keys()).union(faiss_scores.keys())

    data = []
    for doc_id in all_doc_ids:
        bm25_detail = bm25_scores.get(doc_id, {})
        bm25_score = bm25_detail.get('score', 0) if isinstance(bm25_detail, dict) else bm25_detail
        faiss_score = faiss_scores.get(doc_id, 0)
        rrf_score = rrf_scores.get(doc_id, 0) if isinstance(rrf_scores.get(doc_id, 0),
                                                            (int, float)) else rrf_scores.get(doc_id, {}).get('score',
                                                                                                              0)

        data.append({
            "doc_id": doc_id,
            "BM25 Score": bm25_score,
            "FAISS Distance": faiss_score,
            "RRF Score": rrf_score
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by="RRF Score", ascending=False)
    df.to_excel(file_name, index=False)


def save_scores_to_file(
    scores: Dict[int, Any],
    file_name: str
) -> None:
    """
    Saves scores to a text file, including details about tokens that contributed to each score.

    Args:
        scores: A dictionary containing scores and token details for each document/chunk.
        file_name: The file path where the scores will be saved.
    """
    with open(file_name, "w", encoding='utf-8') as scores_file:
        for doc_id, details in scores.items():
            if isinstance(details, dict):
                overall_score = details.get('score', 0)
                tokens = details.get('tokens', [])
                scores_file.write(f"Chunk ID: {doc_id}, Score: {overall_score}\n")
                if tokens:
                    tokens_formatted = "\n".join(tokens)
                    scores_file.write(f"Tokens that were relevant:\n{tokens_formatted}\n")
                else:
                    scores_file.write("Tokens that were relevant: None\n")
                scores_file.write("\n")
            else:
                f.write(f"Chunk ID: {doc_id}, Score: {details}\n\n")
    print(f"Scores saved to {file_name}")


# def confirm_input(prompt: str) -> str:
#     """
#     Prompt the user, echo back their entry, and ask to confirm.
#     Loop until they confirm with Y/Enter.
#     """
#     while True:
#         entry = input(prompt).strip()
#         print(f"\nYou entered:\n  {entry}\n")
#         confirm = input("Is this correct? (Y/n): ").strip().lower()
#         if confirm in ("", "y", "yes"):
#             return entry
#         print("Okay, let's try again.\n")
#
#
# def create_config():
#     # 1) Load the template dict
#     template = load_config("configs_system_instruction/config_template.json")
#
#     # 2) For every top‐level key except the big instruction blob, ask:
#     filled = {}
#     for key, default in template.items():
#         if key == "system_instruction_response":
#             continue  # handle later
#         prompt = (
#             f"Enter a value for '{key}':\n"
#             f"  [default: {default}]\n"
#             "> "
#         )
#         val = confirm_input(prompt)
#         filled[key] = val if val else default
#
#     # 3) Now handle system_instruction_response:
#     sis = template["system_instruction_response"]
#
#     #   a) find ALL bracketed placeholders
#     raw_placeholders = re.findall(r"\[([^\]]+?)\]", sis)
#     #   b) dedupe, preserving order
#     seen = []
#     for ph in raw_placeholders:
#         if ph not in seen:
#             seen.append(ph)
#
#     #   c) for each placeholder, prompt the user
#     for ph in seen:
#         prompt = (
#             f"Fill in the placeholder:\n"
#             f"  [{ph}]\n"
#             "> "
#         )
#         replacement = confirm_input(prompt)
#         sis = sis.replace(f"[{ph}]", replacement)
#
#     filled["system_instruction_response"] = sis
#
#     # 4) Finally ask for the output filename
#     config_name = confirm_input(
#         "What should the output config file be called (no .json)?\n> "
#     )
#     output_path = f"configs_system_instruction/{config_name}.json"
#
#     # 5) Write it out
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(filled, f, indent=2)
#
#     print(f"\n✅ Saved new config to {output_path}")
#     return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RAG workflow tests for varying gene counts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs_system_instruction/GSEA.json",
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()

    config_dict = load_config(args.config, print_settings=True)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    globals()['config_name'] = config_name
    globals().update(config_dict)

    # Debug: Check data directories at start
    print("\n=== Checking data directories ===")
    dirs_to_check = [
        ('./data/GSEA/external_gene_data', ['.gz', '.gmt']),
        ('./data/PDF', ['.pdf']),
        ('./data/GSEA/genes_of_interest', ['.xlsx', '.txt'])
    ]

    for dir_path, extensions in dirs_to_check:
        if os.path.exists(dir_path):
            all_files = os.listdir(dir_path)
            relevant_files = [f for f in all_files
                              if any(f.endswith(ext) for ext in extensions)]
            print(f"✓ {dir_path}: {len(relevant_files)} relevant files")
            if relevant_files and len(relevant_files) <= 5:
                for f in relevant_files:
                    print(f"    - {f}")
        else:
            print(f"✗ {dir_path}: DIRECTORY NOT FOUND")

    configured_gene_list_file = config_dict.get("gene_list_path")
    gene_list_file: Optional[str] = None

    try:
        gene_list_file = resolve_gene_list_file(configured_gene_list_file)
        print(f"✓ Using genes from {gene_list_file}")
    except FileNotFoundError:
        gene_list_file = None
        description_gene_list, description_gene_count = load_gene_list_from_descriptions()
        if description_gene_count > 0:
            print("✓ Using genes from ./data/GSEA/external_gene_data/gene_descriptions.csv")
        else:
            raise
    print("=" * 40 + "\n")

    total_runs = query_range
    pbar = tqdm(total=total_runs, desc="Starting GSEA runs")

    for iteration in range(1, query_range + 1):
        pbar.set_description(f"GSEA iteration {iteration}/{query_range}")

        if gene_list_file:
            gene_list_string, regulation, num_genes = initialize_gene_list(
                gene_file_path=gene_list_file or "",
                fdr_threshold=fdr_threshold,
                de_filter_option="combined",
            )
        else:
            gene_list_string, num_genes = load_gene_list_from_descriptions()
            regulation = "provided_list"

        print(f"DEBUG: Gene list created with {num_genes} genes")
        if num_genes > 0:
            print(f"  First few genes: {', '.join(gene_list_string.split(', ')[:5])}...")

        data_dir = './data/GSEA/external_gene_data'
        log_dir = './logs'
        log_path = os.path.join(log_dir, 'file_log.json')
        index_path = './database/faiss_index.bin'
        db_path = './database/reference_chunks.db'
        ncbi_json_dir = './data/JSON/'

        print("\nProcessing files in directory...")
        process_files_in_directory(data_dir, ncbi_json_dir)

        print("\nInitializing database and embeddings...")
        conn = initialize_database(db_path=db_path)
        tokenizer, embeddings_model = load_embeddings_model_and_tokenizer()
        embedding_dim = embeddings_model.config.hidden_size

        index = initialize_faiss_index(embedding_dim, index_path)

        print("\nEmbedding documents...")
        embed_documents_and_save(
            index, conn, tokenizer, embeddings_model,
            data_dir, batch_size=batch_size,
            log_path=log_path, index_path=index_path
        )

        # Debug: Check database content after embedding
        print("\n=== Database Status Check ===")
        conn_check = sqlite3.connect(db_path)
        cursor = conn_check.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        print(f"Database has {chunk_count} chunks after embedding")

        if chunk_count == 0:
            print("WARNING: No chunks in database! Checking for issues...")
            print("  - Check if file_log.json exists and delete it to force reprocessing")
            print("  - Ensure data files exist in the expected directories")
            print("  - Check file formats are correct (.gz, .gmt, .pdf)")
            conn_check.close()
            pbar.update()
            continue  # Skip this iteration since we have no data

        # Get sample of chunks to verify content
        cursor.execute("SELECT file_name, COUNT(*) FROM chunks GROUP BY file_name LIMIT 5")
        file_counts = cursor.fetchall()
        print("Sample of files with chunks:")
        for file_name, count in file_counts:
            print(f"  - {file_name}: {count} chunks")
        conn_check.close()
        print("=" * 40 + "\n")

        # Continue with the workflow
        print("Loading FAISS index and building BM25...")
        index = load_faiss_index(embedding_dim, index_path=index_path)
        conn = sqlite3.connect(db_path)

        # Build BM25 with error handling
        try:
            bm25_index, bm25_chunk_ids, bm25_chunk_texts = build_bm25_index(conn)
            print(f"BM25 index built with {len(bm25_chunk_ids)} documents")
        except ZeroDivisionError as e:
            print(f"ERROR building BM25 index: {e}")
            print("This usually means no chunks were found in the database")
            conn.close()
            pbar.update()
            continue

        print("\nGenerating response...")
        generate_response_and_save(
            query,
            gene_list_string,
            conn, index, tokenizer, embeddings_model,
            bm25_index, bm25_chunk_ids,
            weight_faiss, weight_bm25,
            system_instruction_response,
            num_genes
        )

        pbar.update()
        print(f"Completed iteration {iteration}\n")

    pbar.close()
    print("\n=== All runs completed ===")


if __name__ == "__main__":
    main()
