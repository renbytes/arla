#!/usr/bin/env python3
"""
Concatenates the text of specified files and directories across the monorepo,
including root-level configuration files. Edit the settings to control the output.
"""

from pathlib import Path
from typing import List, Sequence

# ─── SETTINGS ──────────────────────────────────────────────────────────────────
# The root of your monorepo (where this script should be run from).
ROOT = "."

# 1. CONFIGURE WHICH PACKAGES/DIRECTORIES TO SEARCH RECURSIVELY
#    The script will look for directories at the root that start with these prefixes.
SEARCH_PREFIXES = [
    "agent-core",
    "agent-engine",
    "agent-sim",
    "agent-concurrent",
    "agent-persist",
    "docs",
    "simulations",
]

# 2. CONFIGURE WHICH ROOT-LEVEL FILES TO INCLUDE
#    These files will be searched for directly in the ROOT directory.
#    Use this for top-level configuration like Dockerfile, Makefile, etc.
INCLUDE_ROOT_FILES = [
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    "pyproject.toml",
    ".env.example",
]

# 3. CONFIGURE WHICH FILE TYPES TO INCLUDE (within SEARCH_PREFIXES directories)
#    e.g., [".py", ".yaml", ".md"]. An empty list includes all file types.
INCLUDE_EXTS = [".py", ".yaml", ".yml", ".json", ".md", ".txt"]

# 4. CONFIGURE DIRS/FILES TO EXCLUDE
#    Any path containing these names will be skipped.
EXCLUDE = [".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".egg-info", ".venv"]

# 5. CONFIGURE OUTPUT
ENCODING = "utf-8"
JOIN_WITH = "\n\n" + "─" * 160 + "\n"  # A more prominent separator between files
OUTPUT_FILE = "all_code_and_configs.txt"  # The name of the concatenated output file.
BANNER_CHAR = "─"
BANNER_WIDTH = 160
# ───────────────────────────────────────────────────────────────────────────────


def _gather_files(
    root: Path,
    prefixes: Sequence[str],
    root_files: Sequence[str],
    include_exts: set[str],
    exclude: set[str],
) -> List[Path]:
    """Finds all target files based on the configuration."""
    files: List[Path] = []

    # 1. Gather specified root-level files first
    for filename in root_files:
        path = root / filename
        if path.is_file():
            files.append(path)
        else:
            print(f"Warning: Root file not found: {filename}")

    # 2. Find all package dirs at the root that match the prefixes
    package_dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and any(d.name.startswith(p) for p in prefixes)
    ]

    # 3. Recursively find all files in the target package directories
    for base in package_dirs:
        for path in base.rglob("*"):
            # Skip if any part of the path is in the exclude list
            if any(part in exclude for part in path.relative_to(root).parts):
                continue
            # Check if the file has an allowed extension
            if path.is_file() and (
                not include_exts or path.suffix.lower() in include_exts
            ):
                files.append(path)

    # Return a sorted and unique list of files
    return sorted(list(set(files)), key=lambda p: p.relative_to(root).as_posix())


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
    found_files = _gather_files(
        root, SEARCH_PREFIXES, INCLUDE_ROOT_FILES, include_exts, exclude
    )

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
