import imageio.v2 as imageio
import argparse
from pathlib import Path


def create_gif(input_dir: Path, output_file: Path, fps: int):
    """Creates a GIF from a directory of PNG frames."""
    print(f"Searching for frames in: {input_dir}")
    frames = sorted(input_dir.glob("frame_*.png"))

    if not frames:
        print("Error: No frames found.")
        return

    print(f"Found {len(frames)} frames. Creating GIF...")

    with imageio.get_writer(output_file, mode="I", fps=fps, loop=0) as writer:
        for i, frame_path in enumerate(frames):
            image = imageio.imread(frame_path)
            writer.append_data(image)
            print(f"  Processed frame {i + 1}/{len(frames)}", end="\r")

    print(f"\n✅ GIF saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from simulation frames.")
    parser.add_argument("input_dir", type=str, help="Directory containing PNG frames.")
    parser.add_argument("output_file", type=str, help="Path to save the output GIF.")
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for the GIF."
    )
    args = parser.parse_args()

    create_gif(Path(args.input_dir), Path(args.output_file), args.fps)
