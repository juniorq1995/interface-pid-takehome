"""High-level P&ID PDF -> NetworkX graph pipeline."""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.component_types import SHAPE_TO_TYPE, project_coarse_category, type_from_tag
from src.pid_extraction.connector_detection import (
    detect_connections,
    detect_connections_from_vector_segments,
)
from src.pid_extraction.equipment_label_discovery import (
    find_equipment_label_candidates,
    find_tag_for_known_equipment_shape,
)
from src.pid_extraction.graph_builder import build_graph
from src.pid_extraction.llm_ocr_assist import read_tag_with_llm
from src.pid_extraction.ocr_tagging import extract_tag
from src.pid_extraction.pdf_to_image import pdf_to_images
from src.pid_extraction.shape_detection import detect_shapes
from src.pid_extraction.symbol_type_classifier import classify_symbol_type_with_llm
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


# Real bug found and fixed this session: _crop_label_region's margin-based
# crop (ocr_tagging.py) always spans the shape's own full bbox plus a small
# margin -- fine for typical instrument/valve shapes (bbox <=105px on this
# document), but for large equipment/vessel shapes (F-715A/B: 101x437px,
# V-745: 270x56px) that crop balloons wide enough to sweep in a *neighboring*
# symbol's tag instead of the target's own. Confirmed live: llm_ocr_assist
# read F-715A's crop as "FT-101" and F-715B's as "MV-715B" -- both real,
# nearby tags, neither belonging to the vessel. Because graph_builder.py's
# component_type priority trusts a resolved tag over an already-correct
# SHAPE_TO_TYPE/YOLO-sourced type, those two wrong reads silently reclassified
# both vessels away from "equipment" (measured: page 0 equipment coverage
# 2/2 -> 0/2 with llm_ocr_assist on). Deterministic OCR already safely
# returns None for these same oversized crops (empirically confirmed) --
# only the LLM's "always guess something" behavior turns the oversized-crop
# problem into a silent misclassification. Fix: skip LLM tag-reading (both
# the OCR-assist tag pass and Phase 1 type classification) for any shape
# whose bbox exceeds this size -- its type is already known reliably from
# geometry/vector-signature matching in that case, so a guessed tag can only
# hurt, never help.
_MAX_LABEL_CROP_SHAPE_DIM_PX = 150

# Real bug found while diagnosing the tag-recall gap directly (not
# theorized): checked deterministic OCR against every valve-typed shape on
# page 0 post-fix and found several with bboxes like 48x495px and 87x490px
# -- nearly 500px tall. Every real valve/instrument bbox hand-verified this
# session, across dozens of examples, tops out around 105px in either
# dimension. These oversized shapes are almost certainly malformed/mis-scaled
# YOLO detections (a box swallowing a whole vertical pipe run), not real
# symbols with a real tag to read -- confirmed their OCR output is pure
# garbage ('N\n\n25', 'WW', 'O T', ...), not near-misses. They were silently
# inflating this project's already-disclosed raw over-detection counts (36
# vs. 29 real valves, etc.) with detections that were never going to be real.
# Equipment is exempt -- large equipment (vessels, discovered labels) is
# legitimate; only valve/instrument-coarse shapes get filtered.
_MAX_PLAUSIBLE_VALVE_INSTRUMENT_DIM_PX = 150


def _filter_anomalous_valve_instrument_shapes(shapes, component_type_overrides):
    kept = []
    for shape in shapes:
        geometric_type = component_type_overrides.get(shape.shape_id) or SHAPE_TO_TYPE.get(shape.kind, "unknown")
        coarse = project_coarse_category(geometric_type)
        if coarse in ("valve", "instrument") and max(shape.bbox[2], shape.bbox[3]) > _MAX_PLAUSIBLE_VALVE_INSTRUMENT_DIM_PX:
            continue
        kept.append(shape)
    return kept


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
    use_yolo: bool = False, yolo_weights_path: Path | None = None, symbol_type_assist: bool = False,
    equipment_label_discovery: bool = False,
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
    shapes = _filter_anomalous_valve_instrument_shapes(shapes, component_type_overrides)

    if equipment_label_discovery:
        # Real gap this closes: compound equipment symbols (pumps, exchangers,
        # coolers) have no clean geometric signature -- confirmed live that
        # zero shapes from any detector above fall within 95px of the real
        # P-745 pump. Finds them by label text instead (see
        # equipment_label_discovery.py). Checked only against vessel_shapes
        # (the vector-signature equipment already found), not the full merged
        # list -- valves/instruments legitimately cluster next to equipment
        # and must not suppress a real discovery.
        label_candidates = find_equipment_label_candidates(image, vessel_shapes, start_shape_id=len(shapes))
        for candidate in label_candidates:
            component_type_overrides[candidate.shape_id] = "equipment"
        shapes = shapes + label_candidates

    # equipment_label_region candidates are deliberately large (440px)
    # multi-line context crops, not tight single-tag crops -- extract_tag's
    # rotation+PSM+binarize pipeline is built for the latter and produces
    # confident-looking wrong digit reads on the former (confirmed live:
    # binarized read "E-42" on a region where the real tag is "E-742", 3
    # chars away from a match a human would catch instantly). Because
    # extract_tag returns *something* non-None, it silently wins over the
    # correct wide-crop word search below (equipment_label_discovery loop
    # only runs when tags[...] is still None) -- so these shapes skip
    # extract_tag entirely and go straight to that search.
    tags = {
        shape.shape_id: (extract_tag(image, shape) if shape.kind != "equipment_label_region" else (None, None))
        for shape in shapes
    }

    tag_sources: dict[int, str] = {}

    if equipment_label_discovery:
        # Closes a gap the size guard itself created: F-715A/B (and any other
        # already-known large equipment shape) never gets its own tag read at
        # all, by design -- extract_tag runs unconditionally above and
        # reliably returns None on these oversized crops (safe, but also
        # useless), and the llm_ocr_assist loop below explicitly skips
        # anything over _MAX_LABEL_CROP_SHAPE_DIM_PX. Both are correct
        # protections against the oversized-crop misread bug documented
        # above -- but the net effect is these tags are structurally
        # unreachable, not just hard. find_tag_for_known_equipment_shape
        # reuses the same real-vs-corrupted-separator tolerant search as
        # label discovery, targeted at a shape already known to be equipment,
        # so a wrong read here can only ever supply equipment-prefixed tag
        # text -- it can't misclassify the shape the way the original bug did.
        for shape in shapes:
            if tags[shape.shape_id][0] is not None:
                continue
            geometric_type = component_type_overrides.get(shape.shape_id) or SHAPE_TO_TYPE.get(shape.kind, "unknown")
            if project_coarse_category(geometric_type) != "equipment":
                continue
            if max(shape.bbox[2], shape.bbox[3]) <= _MAX_LABEL_CROP_SHAPE_DIM_PX:
                continue
            found_tag = find_tag_for_known_equipment_shape(image, shape)
            if found_tag is not None:
                tags[shape.shape_id] = (found_tag, found_tag)
                tag_sources[shape.shape_id] = "equipment_label_discovery"
    if llm_ocr_assist:
        # Deterministic OCR is always tried first (fast, free); the LLM is only
        # invoked for shapes it couldn't tag — see llm_ocr_assist.py for why
        # (reads the actual pixels instead of correcting already-mangled text).
        for shape in shapes:
            if tags[shape.shape_id][0] is not None:
                continue
            if max(shape.bbox[2], shape.bbox[3]) > _MAX_LABEL_CROP_SHAPE_DIM_PX:
                continue
            llm_tag, llm_raw = read_tag_with_llm(image, shape)
            if llm_tag is not None:
                tags[shape.shape_id] = (llm_tag, llm_raw)
                tag_sources[shape.shape_id] = "llm_assisted"

    if symbol_type_assist:
        # Phase 1 (README "Next: Phase 1"): for shapes that would otherwise
        # reach graph_builder.py as component_type "unknown" -- no tag-resolved
        # type, no YOLO override, and a raster/vector `kind` outside
        # SHAPE_TO_TYPE's 4-entry vocabulary -- ask the local vision model what
        # kind of symbol it actually is. Only fills gaps: never runs for a
        # shape that already has a tag-resolved or YOLO-resolved type, and a
        # classification that comes back None/off-vocabulary just leaves the
        # shape "unknown", same as if this flag were off.
        for shape in shapes:
            if type_from_tag(tags[shape.shape_id][0] or ""):
                continue
            if shape.shape_id in component_type_overrides:
                continue
            if SHAPE_TO_TYPE.get(shape.kind, "unknown") != "unknown":
                continue
            if max(shape.bbox[2], shape.bbox[3]) > _MAX_LABEL_CROP_SHAPE_DIM_PX:
                continue
            coarse, subtype, _raw = classify_symbol_type_with_llm(image, shape)
            resolved = subtype or coarse
            if resolved:
                component_type_overrides[shape.shape_id] = resolved

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
    yolo_weights_path: Path | None = None, symbol_type_assist: bool = False, equipment_label_discovery: bool = False,
) -> nx.Graph:
    images = pdf_to_images(pdf_path, dpi=dpi)
    if page >= len(images):
        raise ValueError(f"Requested page {page}, PDF only has {len(images)} page(s)")
    return _extract_page_graph(
        images[page], pdf_path, page, dpi, llm_ocr_assist, use_yolo, yolo_weights_path, symbol_type_assist,
        equipment_label_discovery,
    )


def extract_pid_graph_all_pages(
    pdf_path: str | Path, dpi: int = 200, llm_ocr_assist: bool = False, use_yolo: bool = False,
    yolo_weights_path: Path | None = None, symbol_type_assist: bool = False, equipment_label_discovery: bool = False,
) -> nx.Graph:
    """Multi-sheet P&ID sets (e.g. the real Interface assignment PDF) are processed
    page by page and merged — component tags are assumed unique across sheets, which
    held for the real document tested against (see README)."""
    images = pdf_to_images(pdf_path, dpi=dpi)
    combined = nx.Graph()
    for page_index, image in enumerate(images):
        page_graph = _extract_page_graph(
            image, pdf_path, page_index, dpi, llm_ocr_assist, use_yolo, yolo_weights_path, symbol_type_assist,
            equipment_label_discovery,
        )
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
