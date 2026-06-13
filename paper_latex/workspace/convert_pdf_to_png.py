#!/usr/bin/env python3
"""
Convert PDF pages to PNG images for visual review.
Uses pymupdf (fitz) to render PDF pages at 150 DPI.
"""

import os
import sys

try:
    import fitz  # pymupdf
except ImportError:
    print("Error: pymupdf not installed. Install with: pip install pymupdf")
    sys.exit(1)

def pdf_to_png(pdf_path, output_dir, dpi=150):
    """
    Convert each page of a PDF to PNG image.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save PNG files
        dpi: Resolution in DPI (default: 150)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)
    print(f"PDF has {len(doc)} pages")

    # Calculate zoom factor from DPI
    # 72 DPI is the base PDF resolution
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Convert each page
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Render page to image
        pix = page.get_pixmap(matrix=mat)

        # Save as PNG
        output_path = os.path.join(output_dir, f"page_{page_num + 1:03d}.png")
        pix.save(output_path)
        print(f"Saved page {page_num + 1} to {output_path}")

    doc.close()
    print(f"\nConversion complete. {len(doc)} pages saved to {output_dir}/")

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "paper.pdf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "page_images"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    pdf_to_png(pdf_path, output_dir, dpi=150)
