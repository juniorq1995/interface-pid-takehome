"""Assemble detected shapes, OCR tags, and connections into a NetworkX graph."""
from __future__ import annotations

import networkx as nx

from src.component_types import SHAPE_TO_TYPE, type_from_tag
from src.pid_extraction.shape_detection import DetectedShape


def build_graph(
    shapes: list[DetectedShape],
    tags: dict[int, tuple[str | None, str]],
    edges: list[tuple[int, int]],
) -> nx.Graph:
    graph = nx.Graph()

    for shape in shapes:
        tag, raw_text = tags.get(shape.shape_id, (None, ""))
        label = tag or f"UNLABELED-{shape.shape_id}"
        component_type = (tag and type_from_tag(tag)) or SHAPE_TO_TYPE.get(shape.kind, "unknown")
        graph.add_node(
            label,
            shape_id=shape.shape_id,
            tag=tag,
            ocr_raw_text=raw_text,
            symbol_kind=shape.kind,
            component_type=component_type,
            bbox=list(shape.bbox),
            tag_confidence="ocr_matched" if tag else "unresolved",
        )

    id_to_label = {
        data["shape_id"]: label for label, data in graph.nodes(data=True)
    }
    for a, b in edges:
        if a in id_to_label and b in id_to_label:
            graph.add_edge(id_to_label[a], id_to_label[b])

    return graph
