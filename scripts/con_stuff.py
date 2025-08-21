#!/usr/bin/env python3
"""
Finds and concatenates specified source files from a monorepo into a single
output file for analysis. The script is configured by editing the settings
in the `main` function.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Any


def _is_path_excluded(
    path: Path, root_path: Path, exclude_patterns: Sequence[str]
) -> bool:
    """
    Checks if a given path should be excluded based on the configured patterns.

    Args:
        path: The path to check.
        root_path: The root directory of the repository.
        exclude_patterns: A sequence of strings that, if found in any part
            of the relative path, will cause the path to be excluded.

    Returns:
        True if the path should be excluded, False otherwise.
    """
    relative_parts = path.relative_to(root_path).parts
    return any(pattern in relative_parts for pattern in exclude_patterns)


def _find_source_files(
    root_path: Path,
    search_prefixes: Sequence[str],
    exclude_patterns: Sequence[str],
    include_extensions: Sequence[str],
) -> Iterator[Path]:
    """
    Finds and yields all files in the repository that match the criteria.

    This function first identifies the top-level package directories and then
    recursively searches for files within them, applying exclusion and
    extension filters.

    Args:
        root_path: The absolute path to the repository root.
        search_prefixes: A list of prefixes for the top-level directories to include.
        exclude_patterns: A sequence of names; any file or directory containing
            one of these will be skipped.
        include_extensions: A sequence of file extensions to include. If empty,
            all file types are considered.

    Yields:
        A Path object for each file that matches the criteria.
    """
    top_level_dirs = [
        d
        for d in root_path.iterdir()
        if d.is_dir() and d.name.startswith(tuple(search_prefixes))
    ]

    for directory in top_level_dirs:
        for file_path in directory.rglob("*"):
            if file_path.is_dir():
                continue

            if _is_path_excluded(file_path, root_path, exclude_patterns):
                continue

            if not include_extensions or file_path.suffix in include_extensions:
                yield file_path


def _format_file_content(
    file_path: Path, root_path: Path, banner_char: str = "─", banner_width: int = 80
) -> str | None:
    """
    Reads a file's content and formats it with a banner.

    Args:
        file_path: The path to the file to read.
        root_path: The root directory of the repository, for creating a relative path.
        banner_char: The character to use for the banner lines.
        banner_width: The total width of the banner.

    Returns:
        A formatted string containing the banner and file content, or None if
        the file is empty or cannot be read.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            print(f"Skipping empty file: {file_path.relative_to(root_path)}")
            return None
    except Exception as e:
        print(
            f"Skipping file due to read error: {file_path.relative_to(root_path)} ({e})"
        )
        return None

    relative_path_str = file_path.relative_to(root_path).as_posix()
    title = f" FILE: {relative_path_str} "
    padding = banner_char * max(0, banner_width - len(title))
    banner = (
        f"{banner_char * banner_width}\n{title}{padding}\n{banner_char * banner_width}"
    )

    return f"{banner}\n{content}"


def concatenate_repository_files(
    root_path: Path,
    output_file_name: str,
    config: Dict[str, Any],
) -> None:
    """
    Orchestrates the process of finding, formatting, and concatenating files.

    Args:
        root_path: The absolute path to the repository root.
        output_file_name: The name of the file to write the output to.
        config: A dictionary containing all configuration settings.
    """
    output_path = root_path / output_file_name
    all_formatted_parts: List[str] = []

    source_files = sorted(
        _find_source_files(
            root_path=root_path,
            search_prefixes=config.get("search_prefixes", []),
            exclude_patterns=config.get("exclude_patterns", []),
            include_extensions=config.get("include_extensions", []),
        )
    )

    print(f"Found {len(source_files)} files to concatenate...")

    for file_path in source_files:
        formatted_content = _format_file_content(file_path, root_path)
        if formatted_content:
            all_formatted_parts.append(formatted_content)

    final_content = "\n\n".join(all_formatted_parts)
    output_path.write_text(final_content, encoding="utf-8")
    print(f"\n✅ Wrote concatenated output to {output_path.relative_to(root_path)}")


def main() -> None:
    """
    Defines the configuration and runs the script.
    """
    # ─── SETTINGS ──────────────────────────────────────────────────────────
    config = {
        "search_prefixes": [
            "agent-core",
            "agent-engine",
            "agent-sim",
            "agent-concurrent",
            "agent-persist",
            "docs",
            "simulations",
        ],
        "include_extensions": [
            ".py",
            ".yaml",
            ".yml",
            ".json",
            "Dockerfile",
            ".md",
            ".txt",
        ],
        "exclude_patterns": [
            ".git",
            "__pycache__",
            ".ruff_cache",
            ".pytest_cache",
            ".venv",
            ".egg-info",
        ],
    }
    # ───────────────────────────────────────────────────────────────────────

    root_path = Path(".").resolve()
    output_file_name = "all_code.txt"
    concatenate_repository_files(root_path, output_file_name, config)


if __name__ == "__main__":
    main()
