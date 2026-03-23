import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "mghuibregtse/biolinkbert-large-simcse-rat"
DEFAULT_CONFIG_PATH = "./configs_system_instruction/GSEA.json"
DEFAULT_GMT_PATH = "./copy_rat_external_gene_data/wikipathways_synonyms_Rattus_norvegicus.gmt"
DEFAULT_TARGET_PATHWAY = "Irinotecan pathway"

"""
Mapping to pipeline implementation in RAG_workflow.py:
- Model source: config key `embeddings_model_name` (load_config defaults).
- Pathway/document embedding logic mirrors `embed_documents`:
    tokenizer(..., truncation=True, padding=True, max_length=512)
    + masked mean pooling over last_hidden_state.
- Query embedding logic mirrors `query_faiss_index`:
    outputs.last_hidden_state.mean(dim=1) (unmasked mean pooling).
- Similarity logic mirrors FAISS cosine/IP usage in pipeline:
    L2-normalize vectors, then compute dot-product similarity.
"""

def parse_gmt_pathways(gmt_path: str) -> Dict[str, List[str]]:
    """Parse a GMT file and return pathway -> representative gene symbols.

    For synonym-style GMT entries like `[GeneA, alias1, alias2]`, the first token
    is used as representative (e.g., `GeneA`).
    """
    path = Path(gmt_path)
    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {gmt_path}")

    pathways: Dict[str, List[str]] = {}
    bracket_pattern = re.compile(r"\[(.*?)\]")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split("\t")
        if len(fields) < 3:
            continue

        pathway_name = fields[0].strip()
        genes_raw = fields[2:]
        genes: List[str] = []
        seen = set()

        for entry in genes_raw:
            entry = entry.strip()
            if not entry:
                continue

            match = bracket_pattern.search(entry)
            content = match.group(1) if match else entry.strip("[]")
            representative = content.split(",")[0].strip()
            if representative and representative not in seen:
                seen.add(representative)
                genes.append(representative)

        if genes:
            pathways[pathway_name] = genes

    if not pathways:
        raise ValueError(f"No valid pathways were parsed from GMT file: {gmt_path}")
    return pathways

def format_gene_set(genes: Sequence[str]) -> str:
    """Convert a gene list to a single text string for embedding."""
    return "Genes: " + ", ".join(genes)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean pooling (same strategy as embed_documents in RAG_workflow.py)."""
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * expanded_mask, dim=1)
    counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """Embed pathway texts like embed_documents in RAG_workflow.py."""
    embeddings: List[torch.Tensor] = []
    model.eval()
    batch_size = 16

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).to(dtype=torch.float32)


def embed_query_text_identical_pipeline(
    query_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """Embed one query exactly like query_faiss_index in RAG_workflow.py."""
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            [query_text],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        outputs = model(**encoded)
        query_embedding = outputs.last_hidden_state.mean(dim=1).cpu()[0].to(dtype=torch.float32)
    return query_embedding


def embed_query_texts_identical_pipeline(
    query_texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> torch.Tensor:
    """Embed query texts using the query-side pooling in query_faiss_index."""
    embeddings: List[torch.Tensor] = []
    model.eval()
    batch_size = 16

    with torch.no_grad():
        for start in range(0, len(query_texts), batch_size):
            batch = list(query_texts[start:start + batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            outputs = model(**encoded)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0).to(dtype=torch.float32)


def cosine_similarity(query_embedding: torch.Tensor, pathway_embeddings: torch.Tensor) -> torch.Tensor:
    """Return IP scores after L2 norm, matching FAISS cosine behavior in RAG_workflow.py."""
    return pathway_embeddings @ query_embedding


def normalize_l2(embeddings: torch.Tensor) -> torch.Tensor:
    """Apply row-wise L2 normalization, same intent as faiss.normalize_L2 in pipeline."""
    norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-12)
    return embeddings / norms


def get_model_name_from_config(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Read embeddings model from config JSON, aligned with load_config defaults in RAG_workflow.py."""
    path = Path(config_path)
    if not path.exists():
        return MODEL_NAME

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except (json.JSONDecodeError, OSError):
        return MODEL_NAME

    return data.get("embeddings_model_name", MODEL_NAME)


def run_test() -> int:
    """Run top-1 retrieval tests on the rat GMT pathway database."""
    parser = argparse.ArgumentParser(description="Embedding retrieval test for pathway gene sets.")
    parser.add_argument(
        "--gmt-path",
        default=DEFAULT_GMT_PATH,
        help="Path to GMT file to index and test.",
    )
    parser.add_argument(
        "--pathway",
        default=DEFAULT_TARGET_PATHWAY,
        help="Pathway name to use as query (default: Irinotecan pathway).",
    )
    parser.add_argument(
        "--all-pathways",
        action="store_true",
        help="Run self-retrieval test for every pathway in the GMT file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked pathways to print for single-pathway mode.",
    )
    args = parser.parse_args()

    gmt_path = args.gmt_path
    pathways = parse_gmt_pathways(gmt_path)

    pathway_items: List[Tuple[str, List[str]]] = list(pathways.items())
    pathway_names = [name for name, _ in pathway_items]
    pathway_texts = [format_gene_set(genes) for _, genes in pathway_items]
    selected_model_name = get_model_name_from_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Loading model: {selected_model_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)
    model = AutoModel.from_pretrained(selected_model_name)
    model.to(device)

    pathway_embeddings = embed_texts(pathway_texts, tokenizer, model, device)
    pathway_embeddings = normalize_l2(pathway_embeddings)

    if args.all_pathways:
        query_embeddings = embed_query_texts_identical_pipeline(pathway_texts, tokenizer, model, device)
        query_embeddings = normalize_l2(query_embeddings)

        scores_matrix = query_embeddings @ pathway_embeddings.T
        top_indices = torch.argmax(scores_matrix, dim=1).tolist()

        correct = 0
        failures: List[Tuple[str, str, float]] = []

        for i, top_idx in enumerate(top_indices):
            expected_pathway = pathway_names[i]
            retrieved_pathway = pathway_names[top_idx]
            top_score = float(scores_matrix[i, top_idx].item())
            if expected_pathway == retrieved_pathway:
                correct += 1
            else:
                failures.append((expected_pathway, retrieved_pathway, top_score))

        total = len(pathway_names)
        accuracy = correct / total

        print("\n=== Full GMT Embedding Self-Retrieval Test ===")
        print(f"GMT file: {gmt_path}")
        print(f"Total pathways tested: {total}")
        print(f"Top-1 self-retrieval: {correct}/{total} ({accuracy:.2%})")

        if failures:
            print("\nExamples of mismatches (up to 10):")
            for expected, retrieved, score in failures[:10]:
                print(f"- expected='{expected}' retrieved='{retrieved}' score={score:.4f}")

        assert correct == total, (
            "Embedding test failed: not all pathways retrieved themselves as top-1. "
            f"Passed {correct}/{total}."
        )
        print("Assertion passed: every pathway retrieved itself as top-1.")
        return 0

    if args.pathway not in pathways:
        available = ", ".join(pathway_names[:10])
        raise ValueError(
            f"Pathway '{args.pathway}' not found in GMT. "
            f"Example available pathways: {available}"
        )

    query_genes = pathways[args.pathway]
    query_text = format_gene_set(query_genes)
    query_embedding = embed_query_text_identical_pipeline(query_text, tokenizer, model, device)
    query_embedding = query_embedding / torch.norm(query_embedding).clamp(min=1e-12)

    scores = cosine_similarity(query_embedding, pathway_embeddings)
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    top_k = max(1, min(args.top_k, len(ranked_indices)))

    print("\n=== Single Pathway A Retrieval Test ===")
    print(f"GMT file: {gmt_path}")
    print(f"Query pathway (expected top-1): {args.pathway}")
    print(f"Query genes count: {len(query_genes)}")
    print(f"Top-{top_k} ranked pathways:")

    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        pathway_name = pathway_names[idx]
        score = float(scores[idx].item())
        print(f"{rank:>2}. {pathway_name} | score={score:.4f}")

    top_pathway = pathway_names[ranked_indices[0]]
    print(f"\nRetrieved top pathway: {top_pathway}")

    assert top_pathway == args.pathway, (
        "Embedding test failed: expected top-1 pathway "
        f"'{args.pathway}', got '{top_pathway}'."
    )
    print("Assertion passed: expected pathway ranked #1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
