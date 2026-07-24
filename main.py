"""CLI: extract a P&ID graph, cross-reference it against an SOP, write a report.

Usage:
    python main.py --pid data/pid/diagram.pdf --sop data/sop/sop.docx --out output/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.crossref.compare import cross_reference
from src.crossref.report import write_graph, write_report
from src.pid_extraction.pipeline import extract_pid_graph
from src.sop_extraction.docx_parser import parse_sop
from src.sop_extraction.tag_extractor import extract_sop_facts


def run(pid_path: Path, sop_path: Path, out_dir: Path) -> None:
    graph = extract_pid_graph(pid_path)
    print(f"P&ID: extracted {graph.number_of_nodes()} components, {graph.number_of_edges()} connections")

    sop = parse_sop(sop_path)
    facts = extract_sop_facts(sop.full_text)
    print(
        f"SOP: {len(facts.referenced_tags)} tags referenced, "
        f"{len(facts.declared_types)} typed mentions, {len(facts.stated_connections)} stated connections"
    )

    findings = cross_reference(graph, facts)
    write_graph(graph, out_dir)
    write_report(findings, out_dir)

    errors = sum(1 for f in findings if f.severity == "ERROR")
    warnings = sum(1 for f in findings if f.severity == "WARNING")
    print(f"Cross-reference: {errors} error(s), {warnings} warning(s) -> {out_dir}/cross_reference_report.log")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=Path, default=Path("data/pid/diagram.pdf"))
    parser.add_argument("--sop", type=Path, default=Path("data/sop/sop.docx"))
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args()
    run(args.pid, args.sop, args.out)


if __name__ == "__main__":
    main()
