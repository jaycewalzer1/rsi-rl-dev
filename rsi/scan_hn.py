"""Stage 1b: scan Hacker News for practitioner complaints about coding agents.

Uses the public Algolia HN search API (no key). Comments and story texts both count;
each row keeps its HN permalink. Labelling is the same judge pass as the X scan
(scan_x.label), so both sources land in one complaints.jsonl with `provenance: "hn"`.
"""

from __future__ import annotations

import html
import re
import time

import httpx

QUERIES = [
    "claude code missed edge case",
    "coding agent said tests pass",
    "cursor agent broke regression",
    "llm agent timezone bug",
    "ai agent retry duplicate charge",
    "copilot agent only fixed one place",
    "agent ignored requirement refactor",
    "claude code idempotency",
]
API = "https://hn.algolia.com/api/v1/search_by_date"


def _clean(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def search(query: str, days: int = 365, hits: int = 40) -> list[dict]:
    since = int(time.time()) - days * 86400
    r = httpx.get(API, params={"query": query, "tags": "(comment,story)", "hitsPerPage": hits,
                               "numericFilters": f"created_at_i>{since}"}, timeout=30)
    r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        text = _clean(h.get("comment_text") or h.get("story_text") or h.get("title") or "")
        if len(text) < 60:
            continue
        out.append({
            "id": f"hn{h['objectID']}",
            "url": f"https://news.ycombinator.com/item?id={h['objectID']}",
            "text": text[:1200],
            "created_at": h.get("created_at"),
            "points": h.get("points"),
            "query": query,
            "provenance": "hn",
        })
    return out
