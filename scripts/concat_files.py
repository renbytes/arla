#!/usr/bin/env python3
"""
Concatenates the text of specified files across multiple 'agent-*' packages
in the monorepo. Edit the settings to control the output.
"""

from pathlib import Path
from typing import List, Sequence

# ─── SETTINGS ──────────────────────────────────────────────────────────────────
# The root of your monorepo (where this script should be run from).
ROOT = "."

# 1. CONFIGURE WHICH PACKAGES TO SEARCH
#    The script will look for directories at the root that start with these prefixes.
SEARCH_PREFIXES = ["agent-core", "agent-engine", "agent-sim", "agent-concurrent", "agent-persist", "docs"]

# 2. CONFIGURE WHICH SUB-DIRECTORIES TO INCLUDE
#    Within each found package, only these sub-folders will be searched.
INCLUDE_SUBDIRS = []

# 3. CONFIGURE WHICH FILE TYPES TO INCLUDE
#    e.g., [".py", ".yaml", ".md"]. An empty list includes all file types.
INCLUDE_EXTS = [".py", ".yaml", ".yml", ".json", "Dockerfile", ".md", ".txt"]

# 4. CONFIGURE DIRS/FILES TO EXCLUDE
#    Any path containing these names will be skipped.
EXCLUDE = [".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".egg-info"]

# 5. CONFIGURE OUTPUT
ENCODING = "utf-8"
JOIN_WITH = "\n\n"
OUTPUT_FILE = "all_code.txt"  # The name of the concatenated output file.
BANNER_CHAR = "─"
BANNER_WIDTH = 80
# ───────────────────────────────────────────────────────────────────────────────


def _gather_files(
    root: Path,
    prefixes: Sequence[str],
    include_subdirs: Sequence[str],
    include_exts: set[str],
    exclude: set[str],
) -> List[Path]:
    """Finds all target files based on the configuration.

    If `include_subdirs` is empty, the entire package directory under each
    matching prefix is searched recursively.
    """
    files: List[Path] = []

    # 1. Find all package dirs at the root that match the prefixes
    package_dirs = [d for d in root.iterdir() if d.is_dir() and any(d.name.startswith(p) for p in prefixes)]

    # 2. Determine which directories to search
    if include_subdirs:  # user specified sub-folders
        search_paths: List[Path] = []
        for pkg_dir in package_dirs:
            for subdir_name in include_subdirs:
                path_to_search = pkg_dir / subdir_name
                if path_to_search.is_dir():
                    search_paths.append(path_to_search)
    else:  # empty => search entire package
        search_paths = list(package_dirs)

    # 3. Recursively find all files in the target search paths
    for base in search_paths:
        for path in base.rglob("*"):
            if any(part in exclude for part in path.relative_to(root).parts):
                continue
            if path.is_file() and (not include_exts or path.suffix.lower() in include_exts):
                files.append(path)

    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _banner(rel_path: str) -> str:
    """Creates a standardized banner for each file."""
    title = f" FILE: {rel_path} "
    # Ensure banner isn't shorter than the title itself
    pad_width = max(0, BANNER_WIDTH - len(title))
    line = BANNER_CHAR * BANNER_WIDTH
    return f"{line}\n{title}{BANNER_CHAR * pad_width}\n{line}"


def dump_repo_contents() -> str:
    """Gathers and concatenates the content of all specified files."""
    root = Path(ROOT).resolve()
    include_exts = {e.lower() for e in INCLUDE_EXTS}
    exclude = set(EXCLUDE)

    parts: List[str] = []
    found_files = _gather_files(root, SEARCH_PREFIXES, INCLUDE_SUBDIRS, include_exts, exclude)

    print(f"Found {len(found_files)} files to concatenate...")

    for fp in found_files:
        rel_path = fp.relative_to(root).as_posix()
        header = _banner(rel_path)
        try:
            body = fp.read_text(ENCODING)
            if not body.strip():  # Skip empty or whitespace-only files
                print(f"Skipping empty file: {rel_path}")
                continue
            parts.append(f"{header}\n{body}")
        except Exception as exc:
            print(f"Skipping {rel_path} due to read error: {exc}")
            continue

    return JOIN_WITH.join(parts)


def main() -> None:
    """Main function to run the script."""
    root = Path(ROOT).resolve()
    out_path = root / OUTPUT_FILE

    concatenated_content = dump_repo_contents()
    out_path.write_text(concatenated_content, ENCODING)

    print(f"\n✅ Wrote concatenated output to {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
