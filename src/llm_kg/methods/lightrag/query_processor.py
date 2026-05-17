"""LightRAGQueryProcessor — dual-level keyword extraction via LLM.

Sends the upstream `keywords_extraction` prompt and parses the strict-JSON
response into `{high_level_keywords, low_level_keywords}`. The two lists are
stashed on `ProcessedQuery`:

- `ProcessedQuery.entities` = low-level keywords (specific entities/proper nouns).
- `ProcessedQuery.meta["high_level"]` = high-level keywords (abstract themes).
- `ProcessedQuery.keywords` = both, for callers that want a flat list.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from llm_kg.methods.lightrag.prompts import DEFAULT_LANGUAGE, PROMPTS
from llm_kg.pipeline.stage import PipelineContext
from llm_kg.pipeline.stages.query_processor import ProcessedQuery, QueryProcessor


def _parse_keywords_json(text: str) -> tuple[list[str], list[str]]:
    """Robust-ish JSON parse: strip code fences, extract first {...} blob."""
    s = text.strip()
    # Strip ```json ... ``` if present
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # Grab the first {...} block (greedy to last })
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    try:
        data: dict[str, Any] = json.loads(s)
    except json.JSONDecodeError:
        return [], []
    hl = data.get("high_level_keywords") or []
    ll = data.get("low_level_keywords") or []
    hl = [str(x).strip() for x in hl if str(x).strip()]
    ll = [str(x).strip() for x in ll if str(x).strip()]
    return hl, ll


class LightRAGQueryProcessor(QueryProcessor):
    name: ClassVar[str] = "LightRAGQueryProcessor"

    def __init__(self, language: str = DEFAULT_LANGUAGE) -> None:
        self.language = language

    async def run(self, inp: str, ctx: PipelineContext) -> ProcessedQuery:
        if ctx.llm is None:
            raise RuntimeError("LightRAGQueryProcessor requires ctx.llm to be set")

        examples_str = "\n".join(PROMPTS["keywords_extraction_examples"])
        prompt = PROMPTS["keywords_extraction"].format(
            language=self.language,
            examples=examples_str,
            query=inp,
        )
        resp = await ctx.llm.generate(prompt)
        hl, ll = _parse_keywords_json(resp.text)

        # Fallback per upstream: if both lists are empty and query is short, use the
        # raw query as the low-level keyword.
        if not hl and not ll and len(inp) < 50:
            ll = [inp]

        return ProcessedQuery(
            text=inp,
            keywords=hl + ll,
            entities=ll,                # low-level → "entity-like"
            meta={"high_level": hl},    # high-level surfaced separately for the retriever
        )
