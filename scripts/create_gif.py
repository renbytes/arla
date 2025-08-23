# FILE: scripts/create_gif.py

import argparse
import os
import subprocess
from pathlib import Path

import imageio


def create_gif(render_dir: str, output_file: str, fps: int):
    """
    Creates a GIF from a directory of frames.

    Args:
        render_dir: The directory containing the PNG frames.
        output_file: The path for the output GIF.
        fps: The frames per second for the GIF.
    """
    frame_dir = Path(render_dir)
    if not frame_dir.is_dir():
        print(f"❌ Error: Render directory not found at '{render_dir}'")
        return

    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        print(f"❌ Error: No PNG frames found in '{render_dir}'")
        return

    print(f"🎬 Found {len(frames)} frames. Compiling into GIF...")
    images = [imageio.imread(frame) for frame in frames]

    # Save the initial, uncompressed GIF
    imageio.mimsave(output_file, images, fps=fps)

    initial_size = os.path.getsize(output_file) / 1024  # in KB
    print(f"✅ Initial GIF created at '{output_file}' (Size: {initial_size:.2f} KB)")

    # Now, compress the GIF using gifsicle
    compress_gif(output_file)


def compress_gif(file_path: str):
    """
    Compresses a GIF in-place using the gifsicle command-line tool.

    Args:
        file_path: The path to the GIF file to compress.
    """
    print("✨ Compressing GIF with gifsicle...")
    # The -O3 flag applies the highest level of optimization.
    # The --lossy=80 flag can significantly reduce file size. Adjust as needed.
    command = [
        "gifsicle",
        "-O3",
        "--lossy=80",
        "-o",
        file_path,  # Output file (overwrite)
        file_path,  # Input file
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        compressed_size = os.path.getsize(file_path) / 1024  # in KB
        print(f"✅ GIF successfully compressed! (New Size: {compressed_size:.2f} KB)")
    except FileNotFoundError:
        print("❌ Error: 'gifsicle' command not found.")
        print("   Please ensure gifsicle is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during GIF compression: {e.stderr}")


def main():
    """
    Main function to parse arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Create and compress a GIF from frames."
    )
    parser.add_argument("render_dir", help="Directory containing the PNG frames.")
    parser.add_argument("output_file", help="Path for the final, compressed GIF.")
    parser.add_argument(
        "--fps", type=int, default=15, help="Frames per second for the GIF."
    )
    args = parser.parse_args()

    create_gif(args.render_dir, args.output_file, args.fps)


if __name__ == "__main__":
    main()
