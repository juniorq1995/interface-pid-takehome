"""Write graph and cross-reference outputs to disk."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.crossref.compare import Discrepancy


def write_graph(graph: nx.Graph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = nx.node_link_data(graph, edges="edges")
    (output_dir / "pid_graph.json").write_text(json.dumps(data, indent=2, default=str))

    # GraphML has no null/list types, so give it a copy with those coerced to strings.
    graphml_copy = graph.copy()
    for _, attrs in graphml_copy.nodes(data=True):
        for key, value in attrs.items():
            if value is None:
                attrs[key] = ""
            elif isinstance(value, (list, tuple)):
                attrs[key] = str(value)
    nx.write_graphml(graphml_copy, output_dir / "pid_graph.graphml")


def write_report(findings: list[Discrepancy], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = [{"severity": f.severity, "category": f.category, "message": f.message} for f in findings]
    (output_dir / "cross_reference_report.json").write_text(json.dumps(payload, indent=2))

    counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for f in findings:
        counts[f.severity] += 1

    lines = [
        "P&ID <-> SOP Cross-Reference Report",
        f"Summary: {counts['ERROR']} error(s), {counts['WARNING']} warning(s), {counts['INFO']} info",
        "",
    ]
    for f in findings:
        lines.append(f"[{f.severity}] ({f.category}) {f.message}")
    if not findings:
        lines.append("No discrepancies found.")

    (output_dir / "cross_reference_report.log").write_text("\n".join(lines) + "\n")
