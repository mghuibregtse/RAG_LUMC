import os
import glob
import re
import time
import sys
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import markdown
from RAG_workflow import query_llm, load_config, initialize_gene_list, load_gene_list_from_descriptions, normalize_max_genes_config
from pymed import PubMed
import pymed
from dotenv import load_dotenv
from plotting import normalize_gene

load_dotenv()


log_dir = './logs'
log_file = os.path.join(log_dir, 'validation_logs.json')
os.makedirs(log_dir, exist_ok=True)


def load_validation_logs(path):
    try:
        # utf-8-sig transparently handles files that start with a UTF-8 BOM.
        with open(path, 'r', encoding='utf-8-sig') as f:
            logs = json.load(f)
        if isinstance(logs, list):
            logs = dict(logs)
        if not isinstance(logs, dict):
            return {}
        return logs
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {path}; starting with empty validation log cache.")
        return {}


validation_logs = load_validation_logs(log_file)


def read_latest_llm_output(answer_dir):
    answer_path = os.path.join(answer_dir, "answer.txt")
    if not os.path.isfile(answer_path):
        raise FileNotFoundError(f"No 'answer.txt' found in {answer_dir!r}")
    with open(answer_path, "r", encoding="utf8") as file:
        content = file.read().strip()
    return content, answer_path


def extract_pathways(answer_text):
    pathways = []
    pathway_dict = {}
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    it = iter(lines)
    for line in it:
        if line.endswith(":"):
            pathway = line[:-1].strip()
            pathways.append(pathway)
            genes_line = next(it, "")
            genes = [gene.strip() for gene in genes_line.split(",") if gene.strip()]
            pathway_dict[pathway] = genes
    return pathways, pathway_dict


def validate_pathways(gpt_answer, ground_truth, instruction, generation_model):
    _, pathway_dict = extract_pathways(gpt_answer)
    prompt = (
        "Based on the identified pathways, confirm whether they match the ground truth pathways. "
        "Additionally, indicate if there are any novel pathways not present in the ground truth. "
        "User provided pathways:\n"
        f"{', '.join(pathway_dict.keys())}\n\n"
        "Ground truth pathways:\n"
        f"{ground_truth}"
    )
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]
    answer = query_llm(messages, instruction, prompt, save=False, generation_model=generation_model, query_range=1)
    return answer


def academic_validation(pathways, pathway_dict, academic_instruction,
                        validation_model):
    academic_results = []
    for pathway in tqdm(pathways, desc="Validating pathways"):
        # print(f"Validating pathway: {pathway}")
        genes = pathway_dict[pathway]
        prompt = (
            f"For the biological pathway '{pathway}', validate the involvement of the genes: {', '.join(genes)}. "
            "Check academic databases such as PubMed, Google Scholar, and GeneCards."
        )
        messages = [
            {"role": "system", "content": academic_instruction},
            {"role": "user", "content": prompt}
        ]
        response = query_llm(messages, academic_instruction, prompt, save=False, generation_model=validation_model,
                             query_range=1)
        if response is None:
            response = "No academic validation response returned."
        academic_results.append((pathway, genes, response.strip()))
    return academic_results


pubmed = PubMed(tool="MyTool", email="my@email.address")

total_matches = 0
credible_matches = 0


pattern = re.compile(
    r'\(?\s*'                    # optional opening parenthesis
    r'(\d{4})\s*,\s*'            # 1) year
    r'([^,]+?)\s*,\s*'           # 2) authors
    r'"([^"]+)"\s*,\s*'          # 3) title
    r'([^\)>]+?)\s*'             # 4) journal (up to ')' or '>')
    r'\)?\s*>\s*'                # optional closing parenthesis + '>'
    r'(.+)',                     # 5) rest of summary
    re.MULTILINE
)


def replace_entry(match):
    global total_matches, credible_matches, validation_logs
    total_matches += 1

    year_str = match.group(1).strip()
    authors_str = match.group(2).strip()
    original_title = match.group(3).strip()
    journal_str = match.group(4).strip()
    original_summary = match.group(5).strip()

    # Removes sources from like Twitter (with grok)
    if original_summary.lower().startswith("no evidence found"):
        return "No evidence found."

    # Build cache key
    key = original_title

    if key in validation_logs:
        entry = validation_logs[key]
        entry['count'] += 1
        is_real = entry['is_real']
        citation_str = entry['citation']
    else:
        query = f'"{original_title}"[Title:~0]'
        try:
            results = list(pubmed.query(query, max_results=1))
        except Exception as e:
            print("PubMed query error:", e)
            results = []

        if results:
            art = results[0]
            year = getattr(art.publication_date, 'year', year_str)
            journal = art.journal or journal_str
            title = art.title
            if art.authors:
                first = art.authors[0]
                last = first.get('lastname', '').strip()
                author_label = f"{last} et al." if len(art.authors) > 1 else last
            else:
                author_label = authors_str

            is_real = True
            citation_str = f'({year}, {author_label}, "{title}", {journal})'
        else:
            # No direct match → keep all original values
            year = year_str
            author_label = authors_str
            journal = journal_str
            title = original_title
            is_real = False
            citation_str = f'({year}, {author_label}, "{title}", {journal})'

        validation_logs[key] = {
            'citation': citation_str,
            'count':    1,
            'is_real':  is_real
        }

    sorted_entries = sorted(
        validation_logs.items(),
        key=lambda item: item[1]['count'],
        reverse=True
    )
    with open(log_file, 'w', encoding='utf8') as logf:
        json.dump(sorted_entries, logf, indent=2)

    # Style output
    styled = (
        f'<span style="color:green;" >\n{citation_str}</span>'
        if is_real else
        f'<span style="color:red;" >\n{citation_str}</span>'
    )
    if is_real:
        credible_matches += 1

    time.sleep(1)
    return f"{styled}{original_summary}"


def main():
    answer_dir = "./output/results/"
    ground_truth_file = "./output/results/ground_truth_pathways.txt"
    system_instruction_file = "./configs_system_instruction/system_instruction_comparison_pathways.txt"
    academic_instruction_file = "./configs_system_instruction/system_instruction_academic_validation_test.txt"
    output_directory = "./output/results/validation_and_reporting"
    os.makedirs(output_directory, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run the RAG workflow tests for varying gene counts.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs_system_instruction/GSEA.json",
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()
    config = load_config(args.config, print_settings=False)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    size = 0
    gene_list_file = config.get("gene_list_path")
    max_genes = normalize_max_genes_config(config.get("max_genes"))[0]
    globals()['config_name'] = config_name
    globals().update(config)

    try:
        if not gene_list_file:
            genes_dir = Path("./data/GSEA/genes_of_interest")
            txt_files = sorted([p for p in genes_dir.glob("*.txt")]) if genes_dir.exists() else []
            if txt_files:
                gene_list_file = str(txt_files[0])
            else:
                gene_list_file = "./data/GSEA/genes_of_interest/C3_9w_detab_fixed.txt"

        gene_list_string, _, size = initialize_gene_list(
            max_genes=max_genes,
            gene_file_path=gene_list_file,
            fdr_threshold=config.get("fdr_threshold"),
            de_filter_option="combined",
        )
    except FileNotFoundError:
        gene_list_string, size = load_gene_list_from_descriptions(max_genes=max_genes)

    input_set = {
        normalize_gene(g)
        for g in gene_list_string.split(",")
        if g.strip()
    }

    with open(ground_truth_file, 'r', encoding="utf8") as file:
        ground_truth = file.read().strip()
    with open(system_instruction_file, 'r', encoding="utf8") as file:
        comparison_instruction = file.read().strip()

    if os.path.exists(academic_instruction_file):
        with open(academic_instruction_file, 'r', encoding="utf8") as file:
            academic_instruction = file.read().strip()
    else:
        print("Academic instruction file not found. Exiting program")
        sys.exit(1)

    try:
        llm_output, latest_file = read_latest_llm_output(answer_dir)
    except FileNotFoundError as e:
        print(e)
        return
    _, pathway_dict = extract_pathways(llm_output)
    output_genes = {
        normalize_gene(g)
        for genes in pathway_dict.values()
        for g in genes
        if g.strip()
    }

    total_output = len(output_genes)
    matched = sum(1 for g in output_genes if g in input_set)
    hallucination_perc = (
        (total_output - matched) / total_output * 100.0
        if total_output > 0
        else 0.0
    )

    for i in range(1):
        print("Validating pathways... using g:Profiler")
        comparison_summary = validate_pathways(llm_output, ground_truth,
                                               comparison_instruction, generation_model=generation_model)
        pathways, pathway_dict = extract_pathways(llm_output)

        academic_results = academic_validation(
            pathways, pathway_dict,
            academic_instruction,
            validation_model=validation_model
        )

        base_name = os.path.splitext(os.path.basename(latest_file))[0]
        md_filename = os.path.join(
            output_directory,
            f"validation_{base_name}_{validation_model}_{i}.md"
        )

        processed_results = []
        for pathway, genes, summary in tqdm(academic_results, desc="Processing academic results"):
            new_summary = pattern.sub(replace_entry, summary)
            processed_results.append((pathway, genes, new_summary))

        with open(md_filename, 'w', encoding="utf8") as md:
            md.write(f"# Pathway Validation Report for {base_name}\n\n")
            md.write("## Hallucination statistics\n")
            md.write(f"- **Input gene‐count (size)**: {size}\n")
            md.write(f"- **Total unique output genes**: {total_output}\n")
            md.write(f"- **Matched (non‐hallucinated)**: {matched}\n")
            md.write(f"- **Hallucination percentage**: {hallucination_perc:.2f}%\n\n")

            md.write("## Table of Contents\n")
            toc = [
                ("Credible sources found", "#credible-sources-found"),
                ("Original genes / pathways", "#original-genes--pathways"),
                ("Automated validation of pathways", "#automated-validation-of-pathways"),
                ("g:Profiler comparison summary", "#gprofiler-comparison-summary"),
            ]
            for title, anchor in toc:
                md.write(f"- [{title}]({anchor})\n")
            md.write("\n")

            percent_credible = (credible_matches / total_matches * 100) if total_matches else 0.0
            md.write("## Credible sources found\n")
            md.write(
                f"**{percent_credible:.1f}% credible matches ({credible_matches} out of {total_matches})**\n\n"
            )

            md.write("## Original genes / pathways\n")
            for pathway, genes in pathway_dict.items():
                md.write(f"- **{pathway}**: {', '.join(genes)}\n")
            md.write("\n")

            md.write("## Automated validation of pathways\n")
            for pathway, genes, new_summary in processed_results:
                md.write(f"### {pathway}\n")
                md.write(f"**Genes involved:** {', '.join(genes)}\n\n")
                md.write(f"{new_summary}\n\n")

            md.write("## g:Profiler comparison summary\n")
            md.write(f"{comparison_summary}\n\n")

        print(f"Markdown validation report created: {md_filename}")


if __name__ == "__main__":
    main()
