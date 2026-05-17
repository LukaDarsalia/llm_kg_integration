"""Parser for LightRAG's delimited-tuple extraction output.

Output format (one record per line, fields separated by `tuple_delimiter`):

    entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
    relation<|#|>src<|#|>dst<|#|>keywords<|#|>description
    <|COMPLETE|>

Mirrors LightRAG's `_handle_single_entity_extraction` and
`_handle_single_relationship_extraction` from `operate.py`. Where their parser
silently skips a malformed record, we do too — the goal is robustness against
LLM-format-drift, not strict validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INVALID_TYPE_CHARS = re.compile(r"[\'()<>|/\\]")
_ENTITY_NAME_MAX_LENGTH = 256


@dataclass
class ParsedEntity:
    name: str
    type: str
    description: str


@dataclass
class ParsedRelation:
    src: str
    dst: str
    keywords: list[str]
    description: str
    weight: float = 1.0


@dataclass
class ParseResult:
    entities: list[ParsedEntity] = field(default_factory=list)
    relations: list[ParsedRelation] = field(default_factory=list)


def _normalize_field(s: str) -> str:
    """Strip whitespace and surrounding quotes from a parsed field."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _normalize_entity_name(s: str) -> str:
    n = _normalize_field(s)
    if len(n) > _ENTITY_NAME_MAX_LENGTH:
        n = n[:_ENTITY_NAME_MAX_LENGTH]
    return n


def _normalize_entity_type(s: str) -> str | None:
    t = _normalize_field(s).lower()
    # comma-separated list → take first token
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
        t = parts[0] if parts else ""
    if not t or _INVALID_TYPE_CHARS.search(t):
        return None
    return t


def parse_extraction(
    text: str,
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
) -> ParseResult:
    """Parse one LLM extraction response. Malformed records are skipped silently."""
    result = ParseResult()
    if not text:
        return result

    # Strip completion delimiter (case-insensitive) and split into records by newline
    pruned = re.sub(re.escape(completion_delimiter), "\n", text, flags=re.IGNORECASE)
    # Handle the LightRAG known-bug case where the LLM uses `<|#|>entity<|#|>` as a
    # separator instead of newlines.
    pruned = pruned.replace(
        f"{tuple_delimiter}entity{tuple_delimiter}",
        f"\nentity{tuple_delimiter}",
    )
    for kw in ("relation", "relationship"):
        pruned = pruned.replace(
            f"{tuple_delimiter}{kw}{tuple_delimiter}",
            f"\n{kw}{tuple_delimiter}",
        )

    for raw in pruned.split("\n"):
        line = raw.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(tuple_delimiter)]
        if not fields:
            continue
        kind = fields[0].lower().strip()

        if kind == "entity" and len(fields) >= 4:
            name = _normalize_entity_name(fields[1])
            etype = _normalize_entity_type(fields[2])
            desc = _normalize_field(fields[3])
            if not name or etype is None:
                continue
            result.entities.append(ParsedEntity(name=name, type=etype, description=desc))

        elif kind in ("relation", "relationship") and len(fields) >= 5:
            src = _normalize_entity_name(fields[1])
            dst = _normalize_entity_name(fields[2])
            if not src or not dst or src == dst:  # drop self-loops
                continue
            kw_raw = _normalize_field(fields[3])
            keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            desc = _normalize_field(fields[4])
            # If a 6th field looks like a float, use it as weight (LightRAG quirk)
            weight = 1.0
            if len(fields) >= 6:
                try:
                    weight = float(_normalize_field(fields[5]))
                except ValueError:
                    pass
            result.relations.append(
                ParsedRelation(
                    src=src, dst=dst, keywords=keywords, description=desc, weight=weight
                )
            )

    return result
