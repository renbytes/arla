# FILE: scripts/compress_blog_images.py
"""
A utility script to find and compress images for the ARLA blog.

This script scans the `docs/blog/assets` directory for common image formats
(JPEG, PNG) and creates web-optimized versions. It resizes images to a
maximum width while maintaining aspect ratio and saves them with a lower
quality setting to reduce file size.

This helps improve the blog's page load speed without significantly
impacting visual quality.
"""

from pathlib import Path
from PIL import Image

# --- Configuration ---
# The target directory where your original blog images are stored.
SOURCE_DIR = "docs/blog/assets"
# The maximum width for compressed images. Larger images will be resized.
MAX_WIDTH = 1024
# The quality setting for saving JPEGs (0-95). Lower is smaller file size.
JPEG_QUALITY = 85


def find_images(directory: str) -> list[Path]:
    """
    Finds all JPEG and PNG images in a given directory.

    Args:
        directory: The path to the directory to scan.

    Returns:
        A list of Path objects for each found image.
    """
    source_path = Path(directory)
    if not source_path.is_dir():
        print(f"❌ Error: Source directory not found at '{directory}'")
        return []

    print(f"🔎 Scanning for images in '{source_path}'...")
    image_extensions = [".jpg", ".jpeg", ".png"]
    return [
        p
        for p in source_path.glob("**/*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]


def compress_image(image_path: Path, max_width: int, quality: int):
    """
    Resizes and compresses a single image, overwriting the original.

    Args:
        image_path: The Path object of the image to process.
        max_width: The maximum width for the output image.
        quality: The quality setting for JPEG compression.
    """
    try:
        with Image.open(image_path) as img:
            # Only resize if the image is wider than the max width
            if img.width > max_width:
                aspect_ratio = img.height / img.width
                new_height = int(max_width * aspect_ratio)
                print(
                    f"  - Resizing '{image_path.name}' from {img.width}px to {max_width}px wide..."
                )
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

            # Save with optimized settings
            print(f"  - Compressing '{image_path.name}'...")
            img.save(
                image_path,
                "JPEG" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "PNG",
                quality=quality,
                optimize=True,
            )
    except Exception as e:
        print(f"❌ Error processing '{image_path.name}': {e}")


def main():
    """
    Main function to orchestrate the image compression process.
    """
    print("--- Starting Blog Image Compression ---")
    images_to_process = find_images(SOURCE_DIR)

    if not images_to_process:
        print("✅ No new images to compress.")
        return

    for image_path in images_to_process:
        compress_image(image_path, MAX_WIDTH, JPEG_QUALITY)

    print(f"\n✅ Done. Processed {len(images_to_process)} images.")
    print("------------------------------------")


if __name__ == "__main__":
    main()
