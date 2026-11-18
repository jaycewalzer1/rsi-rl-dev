"""Stage 2: cross-reference complaints with real papers.

Complaints are grouped by category. For each category we search arXiv and
Semantic Scholar, then a judge model reads the abstracts and keeps only papers
that actually measure or describe that failure. A failure mode survives only
with at least one complaint AND at least one supporting paper. Seed modes
(seeds/failure_modes.json, sourced during the Frist hackathon) are merged in so
the pipeline has a pool even on a quiet day, and so are the modes read out of
this factory's own failed attempts (rsi/mine.py).

Every mode in the pool carries a `tier` saying how good its evidence is, best first:
    measured      solvers slipped on it in this factory (history, or read from rollouts)
    paper+field   a paper measures it and a practitioner reported it
    paper         a paper measures it
    field         a practitioner reported it in full (X API, Hacker News, a blog)
    snippet       only a search-result snippet says so
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import httpx

from . import llm
from .config import CONFIG
from .store import read_json, read_jsonl, write_json

ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = threading.Lock()  # parallel readers share arXiv's one-call-per-3-s limit
SEEDS = Path(__file__).resolve().parent.parent / "seeds" / "failure_modes.json"


def search_arxiv(query: str, n: int = 8) -> list[dict]:
    with _ARXIV:
        return _search_arxiv(query, n)


def _search_arxiv(query: str, n: int) -> list[dict]:
    r = httpx.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": n, "sortBy": "relevance"},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall(f"{ATOM}entry"):
        out.append(
            {
                "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
                "abstract": " ".join(e.findtext(f"{ATOM}summary", "").split()),
                "url": e.findtext(f"{ATOM}id", ""),
                "year": e.findtext(f"{ATOM}published", "")[:4],
                "source": "arxiv",
            }
        )
    time.sleep(3)  # arXiv asks for 3 s between calls
    return out


def search_s2(query: str, n: int = 8) -> list[dict]:
    headers = {"x-api-key": CONFIG.s2_api_key} if CONFIG.s2_api_key else {}
    r = httpx.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": n, "fields": "title,abstract,url,year,externalIds"},
        headers=headers,
        timeout=30,
    )
    if r.status_code == 429:
        return []  # unauthenticated pool is shared; arXiv still covers us
    r.raise_for_status()
    return [
        {
            "title": p["title"],
            "abstract": p.get("abstract") or "",
            "url": p.get("url", ""),
            "year": str(p.get("year") or ""),
            "source": "semantic_scholar",
        }
        for p in r.json().get("data", [])
        if p.get("abstract")
    ]


JUDGE = """A failure mode of AI coding agents, as practitioners describe it:
CATEGORY: {category}
COMPLAINTS:
{complaints}

Candidate papers (title, year, abstract):
{papers}

1. Write one precise sentence naming the failure mode these complaints share.
2. For each paper, say whether its abstract MEASURES or DESCRIBES this failure in coding
   agents or LLM code generation (not just mentions the topic).
3. Name the "textbook build" an agent would write for a task that exercises this mode,
   and the hidden case that build fails.

JSON: {{"mode": "...", "supporting": [<indices of supporting papers>],
"textbook_build": "...", "hidden_case": "..."}}"""


def read_category(cat: str, items: list[dict]) -> dict | None:
    """One reader: search papers for a complaint category, keep it only if a paper backs it."""
    query = f"LLM coding agent {cat.replace('-', ' ')} failure"
    papers = search_arxiv(query) + search_s2(query)
    if not papers:
        return None
    verdict = llm.ask_json(
        JUDGE.format(
            category=cat,
            complaints="\n".join(f"- {c['failure']} ({c['url']})" for c in items[:12]),
            papers="\n".join(
                f"[{i}] {p['title']} ({p['year']}): {p['abstract'][:900]}"
                for i, p in enumerate(papers)
            ),
        ),
        CONFIG.judge_model,
    )
    support = [papers[i] for i in verdict.get("supporting", []) if i < len(papers)]
    if not support:
        return None
    return {
        "id": cat,
        "mode": verdict["mode"],
        "textbook_build": verdict.get("textbook_build", ""),
        "hidden_case": verdict.get("hidden_case", ""),
        "complaints": [{"url": c["url"], "text": c["text"][:400], "provenance": c["provenance"]} for c in items],
        "papers": [{k: p[k] for k in ("title", "url", "year", "source")} for p in support],
        "origin": "scan",
    }


def by_category(complaints: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for c in complaints:
        out[c["category"]].append(c)
    return out


def cross_reference(complaints: list[dict]) -> list[dict]:
    return [m for cat, items in by_category(complaints).items() if (m := read_category(cat, items))]


def tier(mode: dict, measured: set[str] = frozenset()) -> str:
    field = any(c.get("provenance") != "web_snippet" for c in mode.get("complaints", []))
    if mode.get("origin") == "rollout" or mode["id"] in measured:
        return "measured"
    if mode.get("papers"):
        return "paper+field" if field else "paper"
    return "field" if field else "snippet"


def merge_seeds(modes: list[dict]) -> list[dict]:
    """The scan's modes, then the seeds, then modes read from this factory's failed attempts;
    each tagged with how good its evidence is."""
    from . import combine, mine  # both import this module's siblings; keep the import local

    seen = {m["id"] for m in modes}
    modes = modes + [{**s, "origin": "seed"} for s in read_json(SEEDS) if s["id"] not in seen]
    seen = {m["id"] for m in modes}
    modes += [m for m in mine.modes() if m["id"] not in seen]
    st = combine.stats()
    measured = {m for m, n in st.slips.items() if n > 0}
    return [{**m, "tier": tier(m, measured)} for m in modes]


def run(out_dir: Path) -> Path:
    complaints = read_jsonl(out_dir / "complaints.jsonl")
    modes = merge_seeds(cross_reference(complaints) if complaints else [])
    path = out_dir / "failure_modes.json"
    write_json(path, modes)
    fresh = sum(m["origin"] == "scan" for m in modes)
    tiers = {t: sum(m["tier"] == t for m in modes) for t in ("measured", "paper+field", "paper", "field", "snippet")}
    print(f"papers: {fresh} modes backed by complaint + paper, {len(modes) - fresh} seeds or mined; evidence {tiers} -> {path}")
    return path
