"""Stage 1: scan X for practitioner complaints about coding agents.

Two sources, recorded per row in `provenance`:
- "x_api": X API v2 recent search (needs X_BEARER_TOKEN). Full post text and URL.
- "web_snippet": no token, so a headless Claude session with WebSearch looks for
  x.com posts. Usually only search-result snippets; x.com itself returns 402 to
  fetchers. Treat these as weaker evidence.

Each complaint is then labelled by a judge model with the concrete failure it
describes (or "none" for venting, praise and jokes, which are dropped).
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from . import llm
from .config import CONFIG
from .store import short_hash, write_jsonl

QUERIES = [
    '("claude code" OR opus OR codex OR cursor) (missed OR forgot OR ignored) (test OR edge case OR requirement) -is:retweet lang:en',
    '("coding agent" OR "ai agent") (broke OR regression OR "only fixed") -is:retweet lang:en',
    '(llm OR agent) ("race condition" OR idempotency OR "retry") (bug OR wrong) -is:retweet lang:en',
    '(agent OR claude OR gpt) ("said it was done" OR "claimed" OR "tests pass") (but OR wrong) -is:retweet lang:en',
    '(agent OR llm) (timezone OR DST OR "off by one") (bug OR wrong) -is:retweet lang:en',
    '(agent OR copilot OR claude) (authorization OR "access control" OR tenant) (bug OR leak) -is:retweet lang:en',
]

LABEL_PROMPT = """You label practitioner complaints about AI coding agents.

For each post below, decide whether it describes a CONCRETE, REPRODUCIBLE kind of
coding failure (e.g. "fixed the sync path but not the async one", "in-process lock
that breaks with two server copies"). Venting, jokes, praise, pricing and outages
are "none".

Return a JSON list, one object per post, in order:
{"id": <id>, "failure": "<one sentence naming the failure, or none>",
 "category": "<short kebab-case slug, e.g. sibling-path-missed, cross-process-state,
 derived-surface-omitted, timezone-math, authz-by-id, early-stop, other>"}

Posts:
"""


def search_x_api(query: str, max_results: int = 50) -> list[dict]:
    r = httpx.get(
        "https://api.x.com/2/tweets/search/recent",
        params={
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,author_id",
        },
        headers={"Authorization": f"Bearer {CONFIG.x_bearer_token}"},
        timeout=30,
    )
    if r.status_code == 429:
        reset = int(r.headers.get("x-rate-limit-reset", time.time() + 60))
        raise RuntimeError(f"X rate limit, resets at {reset}")
    r.raise_for_status()
    out = []
    for t in r.json().get("data", []):
        out.append(
            {
                "id": t["id"],
                "url": f"https://x.com/i/web/status/{t['id']}",
                "text": t["text"],
                "created_at": t.get("created_at"),
                "likes": t.get("public_metrics", {}).get("like_count", 0),
                "query": query,
                "provenance": "x_api",
            }
        )
    return out


def search_web_snippets(topic_hint: str) -> list[dict]:
    prompt = f"""Use WebSearch to find recent (last 12 months) posts on x.com or twitter.com
where developers complain about a specific way AI coding agents fail. Focus: {topic_hint}.
Run several searches (e.g. site:x.com plus keywords). Do not invent posts.
Return a JSON list of up to 15 objects: {{"url": "...", "text": "<the post text or snippet
exactly as shown>", "is_snippet": true|false}}. Only include URLs you actually saw."""
    rows = llm.ask_json(prompt, CONFIG.judge_model, tools=["WebSearch"])
    out = []
    for row in rows:
        if not row.get("url") or not row.get("text"):
            continue
        out.append(
            {
                "id": short_hash(row["url"]),
                "url": row["url"],
                "text": row["text"],
                "query": topic_hint,
                "provenance": "web_snippet",
            }
        )
    return out


def label(posts: list[dict], batch: int = 25) -> list[dict]:
    labelled = []
    for i in range(0, len(posts), batch):
        chunk = posts[i : i + batch]
        body = "\n".join(f"[{p['id']}] {p['text'][:600]!r}" for p in chunk)
        labels = {str(x["id"]): x for x in llm.ask_json(LABEL_PROMPT + body, CONFIG.judge_model)}
        for p in chunk:
            lab = labels.get(str(p["id"]), {})
            if lab.get("failure", "none").lower() != "none":
                labelled.append({**p, "failure": lab["failure"], "category": lab.get("category", "other")})
    return labelled


def run(out_dir: Path) -> Path:
    posts: dict[str, dict] = {}
    if CONFIG.x_bearer_token:
        for q in QUERIES:
            for p in search_x_api(q):
                posts.setdefault(p["id"], p)
    else:
        print("scan_x: no X_BEARER_TOKEN, falling back to web search snippets (weaker evidence)")
        for hint in [
            "agents that fix one code path and miss its sibling",
            "agents that break concurrency, retries or idempotency",
            "agents that claim tests pass or stop early",
            "agents that get authorization, tenancy or timezone logic wrong",
        ]:
            for p in search_web_snippets(hint):
                posts.setdefault(p["id"], p)
    labelled = label(list(posts.values()))
    path = out_dir / "complaints.jsonl"
    write_jsonl(path, labelled)
    print(f"scan_x: {len(posts)} posts, {len(labelled)} with a concrete failure -> {path}")
    return path
