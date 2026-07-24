"""Cross-reference an extracted P&ID graph against SOP-stated facts."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from src.component_types import coarse_category
from src.sop_extraction.tag_extractor import SopFacts

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"


@dataclass
class Discrepancy:
    severity: str
    category: str
    message: str


def _graph_tags(graph: nx.Graph) -> dict[str, dict]:
    return {
        data["tag"]: data
        for _, data in graph.nodes(data=True)
        if data.get("tag")
    }


def cross_reference(graph: nx.Graph, sop_facts: SopFacts) -> list[Discrepancy]:
    findings: list[Discrepancy] = []
    graph_tags = _graph_tags(graph)

    for tag in sorted(sop_facts.referenced_tags):
        if tag not in graph_tags:
            findings.append(
                Discrepancy(
                    SEVERITY_ERROR,
                    "MISSING_IN_PID",
                    f"SOP references {tag}, but no matching component was detected in the P&ID.",
                )
            )

    for tag in sorted(graph_tags):
        if tag not in sop_facts.referenced_tags:
            findings.append(
                Discrepancy(
                    SEVERITY_INFO,
                    "MISSING_IN_SOP",
                    f"P&ID component {tag} is not referenced anywhere in the SOP.",
                )
            )

    for tag, declared_type in sorted(sop_facts.declared_types.items()):
        node_data = graph_tags.get(tag)
        if node_data is None:
            continue  # already reported as MISSING_IN_PID
        actual = coarse_category(node_data.get("component_type"))
        if actual and actual != declared_type:
            findings.append(
                Discrepancy(
                    SEVERITY_WARNING,
                    "TYPE_MISMATCH",
                    f"SOP describes {tag} as a {declared_type}, but the P&ID symbol was classified as {actual}.",
                )
            )

    for a, b in sorted(sop_facts.stated_connections):
        if a not in graph_tags or b not in graph_tags:
            continue  # already reported as MISSING_IN_PID
        if not graph.has_edge(a, b):
            findings.append(
                Discrepancy(
                    SEVERITY_ERROR,
                    "CONNECTION_MISMATCH",
                    f"SOP states {a} connects to {b}, but no such connection was detected in the P&ID.",
                )
            )

    for label, data in graph.nodes(data=True):
        if data.get("tag") is None:
            findings.append(
                Discrepancy(
                    SEVERITY_WARNING,
                    "UNRESOLVED_TAG",
                    f"Detected {data.get('symbol_kind')} symbol at bbox={data.get('bbox')} "
                    f"could not be OCR-matched to a tag (raw text: {data.get('ocr_raw_text')!r}).",
                )
            )

    severity_order = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}
    findings.sort(key=lambda f: severity_order[f.severity])
    return findings
