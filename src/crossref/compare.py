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


def _graph_tags(graph: nx.Graph) -> dict[str, list[tuple[str, dict]]]:
    """tag -> list of (node_label, node_data), preserving every node that
    resolved to that tag rather than collapsing duplicates (see DUPLICATE_TAG
    below and graph_builder.py's label-disambiguation comment)."""
    result: dict[str, list[tuple[str, dict]]] = {}
    for label, data in graph.nodes(data=True):
        tag = data.get("tag")
        if tag:
            result.setdefault(tag, []).append((label, data))
    return result


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

    for tag, entries in sorted(graph_tags.items()):
        if len(entries) > 1:
            labels = ", ".join(label for label, _ in entries)
            findings.append(
                Discrepancy(
                    SEVERITY_WARNING,
                    "DUPLICATE_TAG",
                    f"{len(entries)} distinct P&ID symbols all resolved to tag {tag} ({labels}) — "
                    f"likely an OCR/extraction ambiguity, or a genuine duplicate tag on the drawing; verify manually.",
                )
            )

    for tag, declared_type in sorted(sop_facts.declared_types.items()):
        entries = graph_tags.get(tag)
        if not entries:
            continue  # already reported as MISSING_IN_PID
        actuals = {coarse_category(data.get("component_type")) for _, data in entries}
        actuals.discard(None)
        # If any node sharing this tag matches the SOP's declared type, don't flag —
        # avoids a false TYPE_MISMATCH caused purely by the duplicate-tag ambiguity
        # itself (already surfaced separately, above).
        if actuals and declared_type not in actuals:
            actual_str = "/".join(sorted(actuals))
            findings.append(
                Discrepancy(
                    SEVERITY_WARNING,
                    "TYPE_MISMATCH",
                    f"SOP describes {tag} as a {declared_type}, but the P&ID symbol(s) were classified as {actual_str}.",
                )
            )

    for a, b in sorted(sop_facts.stated_connections):
        a_entries, b_entries = graph_tags.get(a), graph_tags.get(b)
        if not a_entries or not b_entries:
            continue  # already reported as MISSING_IN_PID
        connected = any(graph.has_edge(label_a, label_b) for label_a, _ in a_entries for label_b, _ in b_entries)
        if not connected:
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
