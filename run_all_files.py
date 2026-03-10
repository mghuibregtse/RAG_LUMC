#!/usr/bin/env python3
import os
import sys
import platform
import runpy
import multiprocessing as mp
import shutil
import json
import re
from datetime import datetime
from pathlib import Path

mp.set_start_method("spawn", force=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Helper to pick up the config flag
def resolve_config_name(default="GSEA"):
    for arg in sys.argv[1:]:
        if arg.startswith("--") and arg != "--config":
            return arg[2:]
    return default


CONFIG_NAME = resolve_config_name("GSEA")
CONFIG_PATH = f"./configs_system_instruction/{CONFIG_NAME}.json"


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def sanitize_label(value: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def resolve_organism_label(config_path: str) -> str:
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return "unknown"

    direct_fields = ["organism", "species", "taxon", "dataset_organism"]
    for key in direct_fields:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return sanitize_label(value)

    text_fields = [
        str(cfg.get("query", "")),
        str(cfg.get("system_instruction_response", "")),
        str(cfg.get("system_instruction", "")),
    ]
    combined_text = " ".join(text_fields).lower()

    keyword_map = {
        "rattus": "rat",
        "rat": "rat",
        "mus musculus": "mouse",
        "mouse": "mouse",
        "mice": "mouse",
        "homo sapiens": "human",
        "human": "human",
    }
    for keyword, label in keyword_map.items():
        if keyword in combined_text:
            return label

    return "unknown"


def create_snapshot(config_name: str, config_path: str) -> Path:
    organism = resolve_organism_label(config_path)
    snapshot_root = Path("./archive_old_run/snapshots/neurology")
    snapshot_root.mkdir(parents=True, exist_ok=True)

    snapshot_name = f"{sanitize_label(config_name)}_{organism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_dir = snapshot_root / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copy_plan = [
        (Path("./output/test_files"), Path("output/test_files")),
        (Path("./output/results"), Path("output/results")),
        (Path("./output/support"), Path("output/support")),
        (Path("./logs"), Path("logs")),
        (Path(config_path), Path("configs") / Path(config_path).name),
    ]

    copied_items = {}
    for src, rel_dst in copy_plan:
        dst = snapshot_dir / rel_dst
        copied_items[str(rel_dst)] = copy_if_exists(src, dst)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_name": config_name,
        "config_path": config_path,
        "organism": organism,
        "copied_items": copied_items,
    }
    (snapshot_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return snapshot_dir

print(platform.system())
if platform.system() == "Windows":
    import asyncio
    from asyncio import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

print(f"Using config: {CONFIG_NAME}")


def run_module_main(module_name, argv):
    """Temporarily swap in sys.argv, then run module.__main__."""
    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = old_argv


print("Running RAG_workflow.py…")
run_module_main("RAG_workflow", ["--config", CONFIG_PATH])

print("Running gprofiler.py…")
run_module_main("gprofiler",   ["--config", CONFIG_PATH])

print("Running automated_validation.py…")
run_module_main("automated_validation", [])

print("Creating run snapshot…")
try:
    snapshot_path = create_snapshot(CONFIG_NAME, CONFIG_PATH)
    print(f"Snapshot created at: {snapshot_path}")
except Exception as e:
    print(f"Snapshot creation failed: {e}")

print("All done.")
