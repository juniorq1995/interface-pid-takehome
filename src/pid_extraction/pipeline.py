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
from src.pid_extraction.vector_equipment import extract_vessel_symbols
from src.pid_extraction.vector_lines import extract_line_segments
from src.pid_extraction.vector_symbols import extract_circle_symbols
from src.pid_extraction.vector_valves import extract_valve_symbols
from src.pid_extraction import yolo_detector
from src.pid_extraction.yolo_detector import (
    class_name_to_component_type,
    detect_symbols as detect_yolo_symbols,
    weights_available,
)


def _merge_shapes(raster_shapes, *vector_shape_lists, yolo_shapes=()):
    """Vector-detected symbols first (exact geometry, higher trust when both
    approaches overlap), then raster shapes, then trained-model detections last,
    re-numbered into one shape_id space. Returns (shapes, component_type_overrides)
    — the overrides map maps a merged shape_id to the YOLO class's component_type,
    since yolo_shapes' `kind` values (e.g. "gate_valve") aren't in the small
    heuristic SHAPE_TO_TYPE vocabulary graph_builder falls back to."""
    merged = []
    component_type_overrides: dict[int, str] = {}
    ordered = [*[s for lst in vector_shape_lists for s in lst], *raster_shapes]
    for shape in ordered:
        merged.append(
            type(shape)(shape_id=len(merged), kind=shape.kind, bbox=shape.bbox, center=shape.center, contour=shape.contour)
        )
    for shape in yolo_shapes:
        new_id = len(merged)
        merged.append(
            type(shape)(shape_id=new_id, kind=shape.kind, bbox=shape.bbox, center=shape.center, contour=shape.contour)
        )
        component_type_overrides[new_id] = class_name_to_component_type(shape.kind.replace("_", " "))
    return merged, component_type_overrides


def _extract_page_graph(
    image, pdf_path: str | Path | None = None, page_index: int = 0, dpi: int = 200, llm_ocr_assist: bool = False,
    use_yolo: bool = False, yolo_weights_path: Path | None = None,
) -> nx.Graph:
    raster_shapes = detect_shapes(image)
    circle_shapes = extract_circle_symbols(pdf_path, page_index, dpi) if pdf_path is not None else []
    valve_shapes = extract_valve_symbols(pdf_path, page_index, dpi) if pdf_path is not None else []
    vessel_shapes = extract_vessel_symbols(pdf_path, page_index, dpi) if pdf_path is not None else []
    # Read yolo_detector.DEFAULT_WEIGHTS_PATH as a live module attribute here
    # rather than a bound default parameter — Python binds default argument
    # values once at function-definition time, so relying on detect_symbols'
    # own default (or importing the name directly) would silently ignore any
    # later reassignment of yolo_detector.DEFAULT_WEIGHTS_PATH (a real bug
    # this comment exists because of: it caused three checkpoint comparisons
    # this session to silently re-test the same old model).
    resolved_weights_path = yolo_weights_path or yolo_detector.DEFAULT_WEIGHTS_PATH
    yolo_shapes = (
        detect_yolo_symbols(image, resolved_weights_path)
        if use_yolo and weights_available(resolved_weights_path)
        else []
    )
    shapes, component_type_overrides = _merge_shapes(
        raster_shapes, circle_shapes, valve_shapes, vessel_shapes, yolo_shapes=yolo_shapes
    )

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

    return build_graph(shapes, tags, edges, tag_sources, component_type_overrides)


def extract_pid_graph(
    pdf_path: str | Path, page: int = 0, dpi: int = 200, llm_ocr_assist: bool = False, use_yolo: bool = False,
    yolo_weights_path: Path | None = None,
) -> nx.Graph:
    images = pdf_to_images(pdf_path, dpi=dpi)
    if page >= len(images):
        raise ValueError(f"Requested page {page}, PDF only has {len(images)} page(s)")
    return _extract_page_graph(images[page], pdf_path, page, dpi, llm_ocr_assist, use_yolo, yolo_weights_path)


def extract_pid_graph_all_pages(
    pdf_path: str | Path, dpi: int = 200, llm_ocr_assist: bool = False, use_yolo: bool = False,
    yolo_weights_path: Path | None = None,
) -> nx.Graph:
    """Multi-sheet P&ID sets (e.g. the real Interface assignment PDF) are processed
    page by page and merged — component tags are assumed unique across sheets, which
    held for the real document tested against (see README)."""
    images = pdf_to_images(pdf_path, dpi=dpi)
    combined = nx.Graph()
    for page_index, image in enumerate(images):
        page_graph = _extract_page_graph(image, pdf_path, page_index, dpi, llm_ocr_assist, use_yolo, yolo_weights_path)
        # Unresolved shapes are labeled "UNLABELED-{shape_id}", and shape_id restarts
        # at 0 on every page — without this, nx.compose silently merges same-labeled
        # nodes from different pages, undercounting real components. Resolved tags
        # are left alone (assumed unique across sheets, per the docstring above).
        page_graph = nx.relabel_nodes(
            page_graph,
            {n: f"UNLABELED-P{page_index}-{n}" for n in page_graph.nodes if n.startswith("UNLABELED-")},
        )
        combined = nx.compose(combined, page_graph)
    return combined
