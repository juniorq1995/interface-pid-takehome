"""High-level P&ID PDF -> NetworkX graph pipeline."""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.pid_extraction.connector_detection import detect_connections
from src.pid_extraction.graph_builder import build_graph
from src.pid_extraction.ocr_tagging import extract_tag
from src.pid_extraction.pdf_to_image import pdf_to_images
from src.pid_extraction.shape_detection import detect_shapes


def extract_pid_graph(pdf_path: str | Path, page: int = 0) -> nx.Graph:
    images = pdf_to_images(pdf_path)
    if page >= len(images):
        raise ValueError(f"Requested page {page}, PDF only has {len(images)} page(s)")
    image = images[page]

    shapes = detect_shapes(image)
    tags = {shape.shape_id: extract_tag(image, shape) for shape in shapes}
    edges = detect_connections(image, shapes)

    return build_graph(shapes, tags, edges)
