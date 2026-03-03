import csv
import json

input_file = r"C:\Users\misha\RAG_LUMC\mart_export.txt"
genes_json_file = r"C:\Users\misha\RAG_LUMC\data\JSON\genes.json"
ncbi_map_file = r"C:\Users\misha\RAG_LUMC\data\JSON\ncbi_id_to_symbol.json"

genes = []
ncbi_map = {}

with open(input_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ncbi_id = int(row["NCBI gene (formerly Entrezgene) ID"]) if row["NCBI gene (formerly Entrezgene) ID"] else None
        synonyms = row.get("Gene Synonym", "").strip()
        if synonyms:
            synonyms = f"[{synonyms}]"
        gene_entry = {
            "Gene stable ID": row["Gene stable ID"],
            "Gene name": row["Gene name"],
            "Gene description": row["Gene description"],
            "Gene Synonyms": synonyms,
            "NCBI gene (formerly Entrezgene) ID": ncbi_id
        }
        genes.append(gene_entry)
        if ncbi_id:
            ncbi_map[str(ncbi_id)] = row["Gene name"]

with open(genes_json_file, 'w', encoding='utf-8') as f:
    json.dump(genes, f, indent=2, ensure_ascii=False)

with open(ncbi_map_file, 'w', encoding='utf-8') as f:
    json.dump(ncbi_map, f, indent=2, ensure_ascii=False)

print(f"✅ Converted {len(genes)} genes to {genes_json_file} and {ncbi_map_file}")
