"""Extract component tags, declared types, and stated connections from SOP text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.component_types import SOP_TYPE_KEYWORDS, TAG_CORE, TAG_PATTERN

CONNECTION_PHRASES = [
    r"connects? to",
    r"connected to",
    r"feeds? into",
    r"flows? (?:from .*? )?to",
    r"discharges? (?:in)?to",
]
CONNECTION_PATTERN = re.compile(
    rf"({TAG_CORE})\s*(?:{'|'.join(CONNECTION_PHRASES)})\s*({TAG_CORE})",
    re.IGNORECASE,
)

TYPE_KEYWORDS = "|".join(SOP_TYPE_KEYWORDS)
# Two alternatives, each with exactly two groups: (keyword, tag) or (tag, keyword).
TYPE_KEYWORD_PATTERN = re.compile(
    rf"\b(?:({TYPE_KEYWORDS})\s+({TAG_CORE})|({TAG_CORE})\s+\(?({TYPE_KEYWORDS})\)?)\b",
    re.IGNORECASE,
)


@dataclass
class SopFacts:
    referenced_tags: set[str] = field(default_factory=set)
    declared_types: dict[str, str] = field(default_factory=dict)  # tag -> coarse type
    stated_connections: set[tuple[str, str]] = field(default_factory=set)


def extract_sop_facts(text: str) -> SopFacts:
    facts = SopFacts()

    for match in TAG_PATTERN.finditer(text.upper()):
        facts.referenced_tags.add(match.group(1))

    for match in TYPE_KEYWORD_PATTERN.finditer(text):
        kw_first, tag_first, tag_second, kw_second = match.groups()
        keyword, tag = (kw_first, tag_first) if kw_first else (kw_second, tag_second)
        facts.declared_types[tag.upper()] = SOP_TYPE_KEYWORDS[keyword.lower()]

    for match in CONNECTION_PATTERN.finditer(text.upper()):
        a, b = match.group(1), match.group(2)
        if a != b:
            facts.stated_connections.add(tuple(sorted((a, b))))

    return facts
