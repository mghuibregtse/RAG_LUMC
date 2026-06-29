#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CleanupTarget:
    relative_path: str
    description: str

    @property
    def path(self) -> Path:
        return ROOT / self.relative_path


INDEX_TARGETS: List[CleanupTarget] = [
    CleanupTarget("database/reference_chunks.db", "SQLite chunk database"),
    CleanupTarget("database/faiss_index.bin", "FAISS vector index"),
    CleanupTarget("logs/file_log.json", "processed-file log"),
]

RESULT_TARGETS: List[CleanupTarget] = [
    CleanupTarget("logs/validation_logs.json", "validation logs"),
    CleanupTarget("output/test_files", "generated test files"),
    CleanupTarget("output/results", "generated result files"),
    CleanupTarget("output/support", "generated support files"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove cached pipeline state before starting a fresh run. "
            "The default scope clears the retrieval index, FAISS store, and file log."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("index", "results", "all"),
        default="index",
        help=(
            "What to remove: 'index' clears retrieval state, 'results' clears generated "
            "outputs, 'all' does both. Default: index."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    return parser.parse_args()


def resolve_targets(scope: str) -> List[CleanupTarget]:
    if scope == "index":
        return INDEX_TARGETS
    if scope == "results":
        return RESULT_TARGETS
    return INDEX_TARGETS + RESULT_TARGETS


def existing_targets(targets: Iterable[CleanupTarget]) -> List[CleanupTarget]:
    return [target for target in targets if target.path.exists()]


def remove_target(target: CleanupTarget) -> None:
    path = target.path
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def print_targets(title: str, targets: Iterable[CleanupTarget]) -> None:
    print(title)
    for target in targets:
        print(f"  - {target.relative_path}: {target.description}")


def confirm_or_exit(targets: List[CleanupTarget], scope: str) -> None:
    print_targets(f"Cleanup scope '{scope}' will remove:", targets)
    print()
    print(
        "Note: removing retrieval state forces a full re-embedding on the next run, "
        "including mice_genes_consolidated.txt.gz."
    )
    response = input("Proceed? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        print("Aborted. No files were removed.")
        raise SystemExit(1)


def main() -> int:
    args = parse_args()
    targets = resolve_targets(args.scope)
    present_targets = existing_targets(targets)

    if not present_targets:
        print(f"Nothing to remove for scope '{args.scope}'.")
        return 0

    if args.dry_run:
        print_targets(f"Dry run for scope '{args.scope}':", present_targets)
        return 0

    if not args.yes:
        confirm_or_exit(present_targets, args.scope)

    removed_count = 0
    for target in present_targets:
        remove_target(target)
        print(f"Removed {target.relative_path}")
        removed_count += 1

    print()
    print(f"Cleanup finished. Removed {removed_count} path(s).")
    if args.scope in {"index", "all"}:
        print("Next pipeline run will rebuild the retrieval state from scratch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())