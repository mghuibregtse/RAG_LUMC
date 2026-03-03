import os
import pandas as pd
import requests
from RAG_workflow import (
    initialize_gene_list,
    load_config,
    load_gene_list_from_descriptions,
    resolve_gene_excel_path,
)
import argparse
from typing import Any, Dict, List, Optional, Tuple, Set


def query_gprofiler_rest(
    genes_list: List[str],
    organism: str = "rnorvegicus",
    user_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Send a POST request to the g:Profiler REST API to perform gene set enrichment analysis.

    Args:
        genes_list: A list of gene identifiers to query.
        organism: The organism database to use (default: "rnorvegicus").
        user_threshold: The significance threshold (p-value) for returned results (default: 0.05).

    Returns:
        A pandas DataFrame containing the raw JSON results from g:Profiler, with additional
        transformation of the 'intersections' field into human-readable gene names if available.

    Raises:
        requests.HTTPError: If the HTTP request to g:Profiler returns an error status.
        ValueError: If the response JSON cannot be parsed or expected fields are missing.
    """
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]
    payload = {
        "organism": organism,
        "query": genes_list,
        "user_threshold": user_threshold,
        "sources": sources,
        "significance_threshold_method": "g_SCS",
        "no_iea": False,
        "domain_scope": "annotated",
        "output": "json",
        "no_evidences": False
    }
    headers = {"User-Agent": "FullPythonRequest"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Replicate the package's transformation of the intersections field:
    if not payload.get("no_evidences", True):
        meta = data.get("meta", {})
        if meta and "query_metadata" in meta and "genes_metadata" in meta:
            reverse_mappings = {}
            # Build reverse mapping for each query in the metadata
            queries = meta["query_metadata"]["queries"].keys()
            for query in queries:
                mapping = meta["genes_metadata"]["query"][query]["mapping"]
                reverse_mapping = {}
                for k, v in mapping.items():
                    if len(v) == 1:
                        reverse_mapping[v[0]] = k
                    else:
                        for items in v:
                            reverse_mapping[items] = items
                reverse_mappings[query] = reverse_mapping
            # Update each result's intersections to contain gene names
            for result in data["result"]:
                query_id = result["query"]
                if query_id not in meta["genes_metadata"]["query"]:
                    continue
                mapping = reverse_mappings[query_id]
                ens_genes = meta["genes_metadata"]["query"][query_id]["ensgs"]
                gene_names = [mapping.get(gene_id, gene_id) for gene_id in ens_genes]
                # Replace each truthy value in intersections with the corresponding gene name
                result["evidences"] = [ev for ev in result["intersections"] if ev]
                result["intersections"] = [gene for ev, gene in zip(result["intersections"], gene_names) if ev]

    df = pd.DataFrame(data["result"])
    return df


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """
    Recursively flatten a nested list structure into a single flat list of elements.

    Args:
        nested_list: A potentially nested list of items, where items may themselves be lists.

    Returns:
        A flat list containing all non-list elements from the original nested_list, in depth-first order.
    """
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def filter_best_per_parent(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a DataFrame of enrichment results, keep only the best (lowest p-value) term per parent ontology.

    The function expects the DataFrame to contain at least the columns 'p_value', 'parents', and 'native'.
    It sorts rows by ascending p-value, then iterates and keeps the first occurrence of each term whose
    parent has not already been included. If required columns are missing, the input DataFrame is returned unchanged.

    Args:
        df: A pandas DataFrame containing enrichment results, with columns:
            - 'p_value': Numeric p-value of each enrichment term.
            - 'parents': A list of parent term IDs associated with each enrichment term.
            - 'native': The unique term ID for each enrichment result.

    Returns:
        A pandas DataFrame filtered to include only one (best p-value) term per parent. If the required
        columns are not present, returns df without modification.
    """
    required_cols = {'p_value', 'parents', 'native'}
    if not required_cols.issubset(df.columns):
        return df
    df_sorted = df.sort_values(by='p_value', ascending=True).reset_index(drop=True)
    kept_terms = set()
    rows_to_keep = []
    for idx, row in df_sorted.iterrows():
        term_id = row['native']
        parent_ids = row['parents'] if isinstance(row['parents'], list) else []
        if any(parent in kept_terms for parent in parent_ids):
            continue
        rows_to_keep.append(idx)
        kept_terms.add(term_id)
    return df_sorted.loc[rows_to_keep].sort_values(by='p_value', ascending=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG workflow.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs_system_instruction/GSEA.json",
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    globals()['config_name'] = config_name
    globals().update(config)

    print(f"Using config: {config_name}")
    max_genes_value = ""
    for i in max_genes:
        max_genes_value = i

    gene_list_string, num_genes = load_gene_list_from_descriptions()
    regulation = "combined"
    if num_genes > 0:
        print(
            f"Using {num_genes} genes from "
            "./data/GSEA/external_gene_data/gene_descriptions.csv"
        )
    else:
        excel_path = resolve_gene_excel_path()
        gene_list_string, regulation, num_genes = initialize_gene_list(
            max_genes=max_genes_value,
            fdr_threshold=fdr_threshold,
            excel_file_path=excel_path,
        )
    print(f"Number of genes: {num_genes}")

    genes = [gene.strip() for gene_list in gene_list_string.split(',')
             for gene in gene_list.split() if gene.strip()]

    if genes:
        try:
            # Query g:Profiler using the REST API
            df_api = query_gprofiler_rest(genes_list=genes, organism="rnorvegicus", user_threshold=0.05)

            # Filter for significant results (p-value < 0.05)
            df_api_sig = df_api[df_api["p_value"] < 0.05].copy()

            # Filter best results per parent term
            df_api_best = filter_best_per_parent(df_api_sig)

            # Rename columns as desired
            df_api_best.rename(columns={
                "name": "Pathway",
                "native": "annotation term",
                "p_value": "p-value"
            }, inplace=True)

            # Build a 'genes' column from intersections, flattening nested lists if needed
            if "intersections" in df_api_best.columns:
                df_api_best['genes'] = df_api_best['intersections'].apply(
                    lambda x: ', '.join(map(str, flatten_list(x))) if isinstance(x, list) else str(x) if pd.notnull(
                        x) else ''
                )
            else:
                df_api_best['genes'] = ''

            # Final DataFrame with desired columns:
            df_final = df_api_best[["Pathway", "description", "annotation term", "source", "p-value", "genes"]]
            os.makedirs("./output/results", exist_ok=True)
            output_path = "./output/results/ground_truth_pathways.txt"
            df_final[["Pathway", "annotation term"]].to_csv(output_path, sep="\t", index=False)
            print(f"Output written to {output_path}")

        except requests.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
