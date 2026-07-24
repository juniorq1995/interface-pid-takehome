"""High-level P&ID PDF -> NetworkX graph pipeline."""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.pid_extraction.connector_detection import (
    detect_connections,
    detect_connections_from_vector_segments,
)
from src.pid_extraction.graph_builder import build_graph
from src.pid_extraction.llm_ocr_assist import read_tag_with_llm
from src.pid_extraction.ocr_tagging import extract_tag
from src.pid_extraction.pdf_to_image import pdf_to_images
from src.pid_extraction.shape_detection import detect_shapes
from src.pid_extraction.vector_lines import extract_line_segments
from src.pid_extraction.vector_symbols import extract_circle_symbols


def _merge_shapes(raster_shapes, vector_shapes):
    """Vector-detected symbols first (exact geometry, higher trust when both
    approaches overlap), then raster shapes, re-numbered into one shape_id space."""
    merged = []
    for shape in [*vector_shapes, *raster_shapes]:
        merged.append(
            type(shape)(shape_id=len(merged), kind=shape.kind, bbox=shape.bbox, center=shape.center, contour=shape.contour)
        )
    return merged


def _extract_page_graph(
    image, pdf_path: str | Path | None = None, page_index: int = 0, dpi: int = 200, llm_ocr_assist: bool = False
) -> nx.Graph:
    raster_shapes = detect_shapes(image)
    vector_shapes = extract_circle_symbols(pdf_path, page_index, dpi) if pdf_path is not None else []
    shapes = _merge_shapes(raster_shapes, vector_shapes)

    tags = {shape.shape_id: extract_tag(image, shape) for shape in shapes}

    tag_sources: dict[int, str] = {}
    if llm_ocr_assist:
        # Deterministic OCR is always tried first (fast, free); the LLM is only
        # invoked for shapes it couldn't tag — see llm_ocr_assist.py for why
        # (reads the actual pixels instead of correcting already-mangled text).
        for shape in shapes:
            if tags[shape.shape_id][0] is not None:
                continue
            llm_tag, llm_raw = read_tag_with_llm(image, shape)
            if llm_tag is not None:
                tags[shape.shape_id] = (llm_tag, llm_raw)
                tag_sources[shape.shape_id] = "llm_assisted"

    edges = []
    if pdf_path is not None:
        segments = extract_line_segments(pdf_path, page_index, dpi)
        edges = detect_connections_from_vector_segments(segments, shapes)
    if not edges:
        # No vector path data (e.g. a scanned P&ID) or nothing matched — raster fallback.
        edges = detect_connections(image, shapes)

    return build_graph(shapes, tags, edges, tag_sources)


def extract_pid_graph(pdf_path: str | Path, page: int = 0, dpi: int = 200, llm_ocr_assist: bool = False) -> nx.Graph:
    images = pdf_to_images(pdf_path, dpi=dpi)
    if page >= len(images):
        raise ValueError(f"Requested page {page}, PDF only has {len(images)} page(s)")
    return _extract_page_graph(images[page], pdf_path, page, dpi, llm_ocr_assist)


def extract_pid_graph_all_pages(pdf_path: str | Path, dpi: int = 200, llm_ocr_assist: bool = False) -> nx.Graph:
    """Multi-sheet P&ID sets (e.g. the real Interface assignment PDF) are processed
    page by page and merged — component tags are assumed unique across sheets, which
    held for the real document tested against (see README)."""
    images = pdf_to_images(pdf_path, dpi=dpi)
    combined = nx.Graph()
    for page_index, image in enumerate(images):
        combined = nx.compose(combined, _extract_page_graph(image, pdf_path, page_index, dpi, llm_ocr_assist))
    return combined
