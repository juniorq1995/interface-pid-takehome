"""Assemble detected shapes, OCR tags, and connections into a NetworkX graph."""
from __future__ import annotations

import networkx as nx

from src.component_types import SHAPE_TO_TYPE, project_coarse_category, type_from_tag
from src.pid_extraction.shape_detection import DetectedShape


def build_graph(
    shapes: list[DetectedShape],
    tags: dict[int, tuple[str | None, str]],
    edges: list[tuple[int, int]],
    tag_sources: dict[int, str] | None = None,
    component_type_overrides: dict[int, str] | None = None,
) -> nx.Graph:
    """component_type_overrides lets a caller supply a component_type directly
    per shape_id (e.g. the trained YOLO detector's own class prediction),
    bypassing the tag-prefix / SHAPE_TO_TYPE fallback below — that fallback's
    vocabulary is small and specific to raster/vector heuristic shape kinds
    ("circle", "bowtie", ...), not the ~180 specific class names a trained
    model can produce."""
    graph = nx.Graph()
    tag_sources = tag_sources or {}
    component_type_overrides = component_type_overrides or {}
    # nx.Graph nodes are keyed by label — two distinct shapes resolving to the
    # same tag (OCR ambiguity, LLM hallucination, or a genuine duplicate tag on
    # the drawing) would otherwise silently collapse into one node, losing a
    # real physical component. Disambiguate the label for the 2nd+ occurrence
    # while keeping `tag` identical on both, so cross_reference.py can still
    # find and flag the collision (DUPLICATE_TAG) instead of hiding it.
    tag_occurrences: dict[str, int] = {}

    for shape in shapes:
        tag, raw_text = tags.get(shape.shape_id, (None, ""))
        if tag is None:
            label = f"UNLABELED-{shape.shape_id}"
        else:
            seen = tag_occurrences.get(tag, 0)
            tag_occurrences[tag] = seen + 1
            label = tag if seen == 0 else f"{tag} (dup {seen + 1})"
        # Real bug found and fixed this session: a resolved tag used to
        # unconditionally win over an already-known geometric/YOLO type, on
        # the assumption that printed tag text is more authoritative than a
        # shape heuristic. That assumption breaks when the tag itself is
        # wrong -- confirmed live: an oversized/ambiguous crop's LLM-read
        # text ("FT-101" on a vessel that was never a flow transmitter)
        # silently reclassified an already-correctly-typed shape. Geometry
        # is demonstrably the more reliable source on this document (100%
        # category coverage without any tag involved at all -- see README).
        # Fix: a tag-derived type is only trusted when it either agrees with
        # the shape's own geometric coarse category (a legitimate
        # refinement, e.g. circle->instrument, tag "PI 715A"->the finer
        # "pressure_indicator") or when geometry has no opinion at all
        # (kind=="unknown", no override -- exactly Phase 1's target case).
        # A tag whose coarse category *contradicts* geometry is dropped for
        # typing purposes (the raw tag text/tag_confidence are unaffected --
        # this only guards component_type).
        tag_type = tag and type_from_tag(tag)
        geometric_type = component_type_overrides.get(shape.shape_id) or SHAPE_TO_TYPE.get(shape.kind, "unknown")
        if tag_type and geometric_type != "unknown" and project_coarse_category(tag_type) != project_coarse_category(geometric_type):
            component_type = geometric_type
        else:
            component_type = tag_type or geometric_type
        default_confidence = "ocr_matched" if tag else "unresolved"
        graph.add_node(
            label,
            shape_id=shape.shape_id,
            tag=tag,
            ocr_raw_text=raw_text,
            symbol_kind=shape.kind,
            component_type=component_type,
            bbox=list(shape.bbox),
            tag_confidence=tag_sources.get(shape.shape_id, default_confidence),
        )

    id_to_label = {
        data["shape_id"]: label for label, data in graph.nodes(data=True)
    }
    for a, b in edges:
        if a in id_to_label and b in id_to_label:
            graph.add_edge(id_to_label[a], id_to_label[b])

    return graph
