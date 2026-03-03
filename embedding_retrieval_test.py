import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "mghuibregtse/biolinkbert-large-simcse-rat"
DEFAULT_CONFIG_PATH = "./configs_system_instruction/GSEA.json"

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


# def build_synthetic_pathways() -> Dict[str, List[str]]:
#     """Return a small handpicked set of real rat pathways from WikiPathways GMT."""
#     return {
#         "Irinotecan pathway": ["Abcc1", "Abcg2", "Abcc2"],
#         "Glucuronidation": ["Abcc2", "Abcg2", "Ugp2"],
#         "Non homologous end joining": ["Prkdc", "Xrcc6", "Xrcc4"],
#         "EBV LMP1 signaling": ["Nfkb2", "Mapk8", "Abcc1"],
#         "Estrogen signaling": ["Esr1", "Ep300", "Ccnd1"],
#         "Transcriptional activation by Nfe2l2 in response to phytochemicals": ["Nfe2l2", "Hmox1", "Abcg2"],
#         "Methylation": ["Mat1a", "Mat2a", "Abcc1"],
#     }

def build_synthetic_pathways() -> Dict[str, List[str]]:
    """Return a set of human pathways with some overlap."""
    return {
        "Irinotecan pharmacokinetics/transport": ["ABCC1", "ABCG2", "ABCC2"],
        "Glucuronidation (UGT-related)": ["ABCC2", "ABCG2", "UGP2"],
        "Non-homologous end joining (NHEJ)": ["PRKDC", "XRCC6", "XRCC4"],
        "NF-kB signaling (LMP1-like / inflammatory)": ["NFKB2", "MAPK8", "ABCC1"],
        "Estrogen receptor signaling": ["ESR1", "EP300", "CCND1"],
        "NRF2 (NFE2L2) oxidative stress response": ["NFE2L2", "HMOX1", "ABCG2"],
        "One-carbon metabolism / methylation": ["MAT1A", "MAT2A", "ABCC1"],
    }

# def build_synthetic_pathways() -> Dict[str, List[str]]:
#     """Return mouse pathways using genes present in the mouse synonyms GMT."""
#     return {
#         "Glucuronidation": ["Ugp2", "Ugt1a1", "Ugt2a2"],
#         "Non homologous end joining": ["Prkdc", "Xrcc6", "Xrcc4"],
#         "EBV LMP1 signaling": ["Nfkb2", "Mapk8", "Chuk"],
#         "Estrogen signaling": ["Esr1", "Ep300", "Ccnd1"],
#         "Transcriptional activation by Nfe2l2 in response to phytochemicals": ["Nfe2l2", "Hmox1", "Nqo1"],
#         "Methylation": ["Mat1a", "Mat2a", "Comt"],
#         "One carbon metabolism and related pathways": ["Mat1a", "Mat2a", "Mthfr"],
#     }

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
    """Run a minimal synthetic retrieval test with embedding steps mapped to RAG_workflow.py."""
    pathways = build_synthetic_pathways()
    query_genes = ["ABCC1", "ABCG2", "ABCC2"]
    expected_top = "Irinotecan pharmacokinetics/transport"

    pathway_items: List[Tuple[str, List[str]]] = list(pathways.items())
    pathway_names = [name for name, _ in pathway_items]
    pathway_texts = [format_gene_set(genes) for _, genes in pathway_items]
    query_text = format_gene_set(query_genes)
    selected_model_name = get_model_name_from_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Loading model: {selected_model_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(selected_model_name)
    model = AutoModel.from_pretrained(selected_model_name)
    model.to(device)

    pathway_embeddings = embed_texts(pathway_texts, tokenizer, model, device)
    query_embedding = embed_query_text_identical_pipeline(query_text, tokenizer, model, device)

    pathway_embeddings = normalize_l2(pathway_embeddings)
    query_embedding = query_embedding / torch.norm(query_embedding).clamp(min=1e-12)

    scores = cosine_similarity(query_embedding, pathway_embeddings)
    ranked_indices = torch.argsort(scores, descending=True).tolist()

    print("\n=== Synthetic Embedding Similarity Test ===")
    print(f"Query gene set: {query_genes}")
    print("\nRanked pathways by cosine similarity:")

    for rank, idx in enumerate(ranked_indices, start=1):
        pathway_name = pathway_names[idx]
        pathway_genes = pathways[pathway_name]
        score = float(scores[idx].item())
        print(f"{rank:>2}. {pathway_name:<10} score={score:.4f} genes={pathway_genes}")

    top_pathway = pathway_names[ranked_indices[0]]
    print(f"\nExpected top pathway: {expected_top}")
    print(f"Retrieved top pathway: {top_pathway}")

    assert top_pathway == expected_top, (
        f"Embedding test failed: expected top-1 '{expected_top}', got '{top_pathway}'."
    )
    print("Assertion passed: expected pathway ranked #1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
