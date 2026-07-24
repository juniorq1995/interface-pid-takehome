"""Render a P&ID PDF page to a numpy/OpenCV image."""
from __future__ import annotations

from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[np.ndarray]:
    """Render every page of a PDF to a list of BGR numpy images."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"P&ID PDF not found: {pdf_path}")

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise ValueError(f"P&ID PDF has no pages: {pdf_path}")
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            images.append(arr)
    return images
