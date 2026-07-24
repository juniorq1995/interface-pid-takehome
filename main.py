"""CLI: extract a P&ID graph, cross-reference it against an SOP, write a report.

Usage:
    python main.py --pid data/pid/diagram.pdf --sop data/sop/sop.docx --out output/

Multi-sheet PDFs (like the real Interface assignment file) are processed page by
page and merged. See README "Tested Against Real Data" for what this finds on
the real document pair versus the synthetic dev fixture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.crossref.compare import cross_reference
from src.crossref.limit_check import cross_reference_limits
from src.crossref.report import write_graph, write_report
from src.pid_extraction.pipeline import extract_pid_graph_all_pages
from src.pid_extraction.title_block import extract_title_block_limits_all_pages
from src.sop_extraction.docx_parser import parse_sop
from src.sop_extraction.limits_table import extract_equipment_limits
from src.sop_extraction.tag_extractor import extract_sop_facts


def run(pid_path: Path, sop_path: Path, out_dir: Path, llm_ocr_assist: bool = False) -> None:
    graph = extract_pid_graph_all_pages(pid_path, llm_ocr_assist=llm_ocr_assist)
    print(f"P&ID: extracted {graph.number_of_nodes()} components, {graph.number_of_edges()} connections")

    sop = parse_sop(sop_path)
    facts = extract_sop_facts(sop.full_text)
    sop_limits = extract_equipment_limits(sop.tables)
    print(
        f"SOP: {len(facts.referenced_tags)} tags referenced in narrative text, "
        f"{len(sop_limits)} equipment rows in limits table(s)"
    )

    pid_limits = extract_title_block_limits_all_pages(pid_path, known_tags=set(sop_limits.keys()))
    print(f"P&ID title blocks: design limits read for {len(pid_limits)} of {len(sop_limits)} known equipment tags")

    findings = cross_reference(graph, facts) + cross_reference_limits(pid_limits, sop_limits)
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
    parser.add_argument(
        "--llm-ocr-assist",
        action="store_true",
        help="Use a local Ollama vision model to re-read tags Tesseract couldn't. "
        "Slow (~30-90s per unresolved symbol on CPU) — opt-in, off by default. "
        "Requires Ollama running locally with llama3.2-vision:11b pulled.",
    )
    args = parser.parse_args()
    run(args.pid, args.sop, args.out, args.llm_ocr_assist)


if __name__ == "__main__":
    main()
