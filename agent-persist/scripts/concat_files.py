#!/usr/bin/env python3
"""
Concatenate the text of (selected) files in a repo.

Edit the constants below to control what gets included / ignored,
then just run `python dump_repo.py`.
"""

from pathlib import Path
from typing import List, Sequence

# ─── SETTINGS ────────────────────────────────────────────────────────────────
ROOT = "."  # repo root (relative or absolute)
INCLUDE_DIRS = ["src"]  # folders to search; ["src", "tests"] etc.
INCLUDE_EXTS = []  # e.g. [".py", ".md"]; empty → all types
EXCLUDE = [
    ".git",
    "__pycache__",
    "__init__.py",
    ".ruff_cache",
]  # name fragments to skip (dirs or files)
ENCODING = "utf-8"  # assumed text encoding
JOIN_WITH = "\n\n"  # glue between files in the output
OUTPUT_FILE = "all_code"  # output filename (relative to ROOT)
BANNER_CHAR = "─"  # what the banner line is made of
BANNER_WIDTH = 80  # how wide the banner line is
# ─────────────────────────────────────────────────────────────────────────────


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

            if any(part in exclude for part in rel.parts):
                continue
            if path.is_file() and (not include_exts or path.suffix.lower() in include_exts):
                files.append(path)

    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _banner(rel_path: str) -> str:
    title = f" FILE: {rel_path} "
    pad = max(0, BANNER_WIDTH - len(title))
    line = title + (BANNER_CHAR * pad)
    return f"{BANNER_CHAR * BANNER_WIDTH}\n{line}\n{BANNER_CHAR * BANNER_WIDTH}"


def dump_repo_contents() -> str:
    root = Path(ROOT).resolve()
    include_dirs = [root / d for d in INCLUDE_DIRS]
    include_exts = {e.lower() for e in INCLUDE_EXTS}
    exclude = set(EXCLUDE)

    parts: List[str] = []
    for fp in _gather_files(root, include_dirs, include_exts, exclude):
        rel = fp.relative_to(root).as_posix()
        header = _banner(rel)
        try:
            body = fp.read_text(ENCODING)
        except Exception as exc:
            print(f"Skipping {rel}: {exc}")
            continue
        parts.append(f"{header}\n{body}")

    return JOIN_WITH.join(parts)


def main() -> None:
    root = Path(ROOT).resolve()
    out_path = root / OUTPUT_FILE
    out_path.write_text(dump_repo_contents(), ENCODING)
    print(f"Wrote concatenated output to {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
