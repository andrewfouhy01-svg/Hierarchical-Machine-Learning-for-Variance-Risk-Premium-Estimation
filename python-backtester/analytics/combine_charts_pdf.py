"""Combine four chart PDFs into a single-page 2x2 grid (like R's par(mfrow=c(2,2)))."""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from io import BytesIO


CHARTS_2x2 = [
    "equity_curve_drawdown.pdf",   # top-left
    "monthly_returns_heatmap.pdf", # top-right
    "pnl_distribution.pdf",       # bottom-left
    "vix_entry_vs_pnl.pdf",       # bottom-right
]


def _pdf_to_image(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Render first page of a PDF to a numpy RGB array."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:  # RGBA -> RGB
        img = img[:, :, :3]
    doc.close()
    return img


def combine_charts(charts_dir: str | Path, output_path: str | Path | None = None) -> Path:
    charts_dir = Path(charts_dir)
    if output_path is None:
        output_path = charts_dir.parent / "tearsheet_charts.pdf"
    else:
        output_path = Path(output_path)

    images = []
    for name in CHARTS_2x2:
        pdf = charts_dir / name
        if pdf.exists():
            images.append(_pdf_to_image(pdf))
            print(f"  Loaded: {name}")
        else:
            print(f"  NOT FOUND: {name}")
            return output_path

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0.05)

    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.axis("off")

    fig.savefig(str(output_path), format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nCombined 2x2 PDF saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        charts = Path(sys.argv[1])
    else:
        charts = Path(__file__).resolve().parent.parent / "charts"

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    combine_charts(charts, out)
