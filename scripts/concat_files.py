#!/usr/bin/env python3
"""
Concatenate the text of (selected) files in a repo.

Edit the constants below to control what gets included / ignored,
then just run `python dump_repo.py`.
"""

# ─── SETTINGS ────────────────────────────────────────────────────────────────
ROOT = "."  # repo root (relative or absolute)
INCLUDE_DIRS = ["."]  # folders to search; ["src", "tests"] etc.
INCLUDE_EXTS = []  # e.g. [".py", ".md"]; empty → all types
EXCLUDE = [".git", "__pycache__", "__init__.py", ".ruff_cache"]  # name fragments to skip (dirs or files)
ENCODING = "utf-8"  # assumed text encoding
JOIN_WITH = "\n\n"  # glue between files in the output
OUTPUT_FILE = "all_code"  # output filename (relative to ROOT)
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from typing import List, Sequence


def _gather_files(
    root: Path,
    include_dirs: Sequence[Path],
    include_exts: set[str],
    exclude: set[str],
) -> List[Path]:
    files: List[Path] = []
    for base in include_dirs:
        for path in base.rglob("*"):
            rel = path.relative_to(root)

            # Skip anything containing an excluded fragment
            if any(part in exclude for part in rel.parts):
                continue

            if path.is_file():
                if include_exts and path.suffix.lower() not in include_exts:
                    continue
                files.append(path)

    # deterministic ordering for repeatability
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def dump_repo_contents() -> str:
    """Return one big string made of all selected files' contents."""
    root = Path(ROOT).resolve()
    include_dirs = [root / Path(d) for d in INCLUDE_DIRS]
    include_exts = {e.lower() for e in INCLUDE_EXTS}
    exclude = set(EXCLUDE)

    blobs: List[str] = []
    for fp in _gather_files(root, include_dirs, include_exts, exclude):
        try:
            blobs.append(fp.read_text(ENCODING))
        except Exception as exc:
            print(f"Skipping {fp}: {exc}")

    return JOIN_WITH.join(blobs)


def main() -> None:
    root = Path(ROOT).resolve()
    out_path = root / OUTPUT_FILE
    out_path.write_text(dump_repo_contents(), ENCODING)
    print(f"Wrote concatenated output to {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
