"""
Concatenate files from multiple directories into chunks.
This is useful if you're trying to dump the codebase into an LLM for help.

Usage:
    python scripts/maintenance/concat_files.py
"""

import os

import pathspec

# Configuration
target_dirs = ["/Users/bordumb/workspace/repositories/agent-soul-sim/src",
               "/Users/bordumb/workspace/repositories/agent-soul-sim/config",
               ]
extensions = [".py", ".yml", ".yaml"]
chunk_size = 50000  # max lines per chunk
output_dir = "/Users/bordumb/workspace/repositories/agent-soul-sim/all_code/chunks"
ignore_dirs = [
    "all_code",
    "data",
    "venv",
    "__pycache__",
    "analysis",
    "tests",
    "__init__.py"
]


# Load .gitignore spec
def load_gitignore_spec(base_path):
    gitignore_path = os.path.join(base_path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            lines = f.readlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return None


# Collect files respecting .gitignore


def collect_files(directories, extensions, ignore_dirs=None):
    collected = []
    ignore_dirs = set(ignore_dirs or [])
    for base in directories:
        spec = load_gitignore_spec(base)
        for root, dirs, files in os.walk(base):
            # Prevent descending into ignored dirs
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for fname in sorted(files):
                if any(fname.endswith(ext) for ext in extensions):
                    rel = os.path.relpath(os.path.join(root, fname), base)
                    if spec and spec.match_file(rel):
                        continue
                    collected.append(os.path.join(root, fname))
    return collected


# Write chunked files without splitting individual files
def write_chunks(files, max_lines, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    chunk_index = 1
    current_lines = 0
    out_path = os.path.join(out_dir, f"chunk_{chunk_index:03d}.txt")
    out_fp = open(out_path, "w", encoding="utf-8")

    for file_path in files:
        # Count lines in this file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
            continue

        file_line_count = len(lines) + 2  # +2 for header lines

        # If adding this file exceeds chunk size, start a new chunk
        if current_lines + file_line_count > max_lines and current_lines > 0:
            out_fp.close()
            chunk_index += 1
            current_lines = 0
            out_path = os.path.join(out_dir, f"chunk_{chunk_index:03d}.txt")
            out_fp = open(out_path, "w", encoding="utf-8")

        # Write header and content
        out_fp.write(f"\n\n// --- {file_path} ---\n\n")
        out_fp.writelines(lines)
        current_lines += file_line_count

    out_fp.close()
    print(f"✅ Wrote {chunk_index} chunks in {out_dir}")


# Main execution
def main():
    matching = collect_files(target_dirs, extensions, ignore_dirs)
    print(f"✅ Found {len(matching)} matching files")
    write_chunks(matching, chunk_size, output_dir)


if __name__ == "__main__":
    main()
