"""The whole factory as a swarm of agents, logged to runs/<run>/events.jsonl for the demo page.

    intake     loader per table, pruner per table, steward agent per table (judge model)
    scan       HN scouts (Algolia API, one per query) + X scouts (X API, or web-search snippets)
               then labeller agents, 25 posts each, keep concrete failures only
    papers     one reader agent per complaint category: arXiv + Semantic Scholar, judge keeps
               the category only if a paper measures it; seed modes merged in
    crossref   mapper agents tie every task rule (bizops.RULES) to the complaints and papers
               that back it, and the history prior orders rules by how often they made solvers slip
    design     the "impossible" task: every rule stacked (level 8), plus the descending ladder
               (which rule comes off at each step, slip-weakest first)
    export     the impossible task and the ladder steps as Harbor tasks (docs/TASK_CONTRACT.md)
    gate       your pipeline: `rsi gate` shape, oracle, nop, mutations on the exported tasks;
               difficulty/calibrate when solver credentials are set
    qwen       (separate GPU-queue job, `rsi qwen`) benchmark, anneal, post-train, benchmark

    rsi swarm --run NAME [--business PATH] [--skip-scan] [--gate STAGES]
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from . import bizops, events, gate, intake, llm, papers, scan_hn, scan_x
from .combine import mode_weights
from .config import CONFIG
from .store import read_json, write_json, write_jsonl

DEFAULT_BUSINESS = Path(__file__).resolve().parent.parent / "examples" / "business" / "shoe-store" / "raw"
X_HINTS = [
    "agents that fix one code path and miss its sibling",
    "agents that break retries, duplicate events or idempotency",
    "agents that claim tests pass or stop early",
    "agents that get time zones, date windows or off-by-one wrong",
    "agents that leak data across tenants or skip authorization",
]


def scan(out: Path) -> list[dict]:
    events.emit("stage_start", stage="scan", msg="HN + X scouts")
    jobs = [("hn", q) for q in scan_hn.QUERIES]
    if CONFIG.x_bearer_token:
        jobs += [("x_api", q) for q in scan_x.QUERIES]
    else:
        jobs += [("x_web", h) for h in X_HINTS]

    def scout(job, agent):
        kind, q = job
        if kind == "hn":
            rows = scan_hn.search(q)
        elif kind == "x_api":
            rows = scan_x.search_x_api(q)
        else:
            rows = scan_x.search_web_snippets(q)
        for r in rows[:3]:
            events.emit("work", stage="scan", agent=agent, role=kind, msg=r["text"][:140], url=r["url"])
        return rows

    found = events.swarm("scan", "scout", jobs, scout, workers=8, label=lambda j: f"{j[0]}: {j[1]}")
    posts: dict[str, dict] = {}
    for rows in found:
        for r in rows or []:
            posts.setdefault(r["id"], r)
    allp = list(posts.values())
    batches = [allp[i:i + 25] for i in range(0, len(allp), 25)]
    labelled = events.swarm("scan", "labeller", batches, lambda b, a: scan_x.label(b), workers=6,
                            label=lambda b: f"{len(b)} posts")
    complaints = [c for chunk in labelled for c in chunk or []]
    for c in complaints:
        events.emit("artifact", stage="scan", role=c["provenance"], msg=c["failure"], url=c["url"], category=c["category"])
    write_jsonl(out / "complaints.jsonl", complaints)
    by_src = {s: sum(c["provenance"] == s for c in complaints) for s in {c["provenance"] for c in complaints}}
    events.emit("stage_end", stage="scan", msg=f"{len(allp)} posts read, {len(complaints)} concrete failures {by_src}",
                posts=len(allp), complaints=len(complaints))
    return complaints


def read_papers(out: Path, complaints: list[dict]) -> list[dict]:
    events.emit("stage_start", stage="papers", msg="one reader per complaint category")
    cats = papers.by_category(complaints)
    found = events.swarm("papers", "reader", list(cats.items()), lambda kv, a: papers.read_category(*kv),
                         workers=6, label=lambda kv: f"{kv[0]} ({len(kv[1])} complaints)")
    modes = papers.merge_seeds([m for m in found if m])
    for m in modes:
        for p in m.get("papers", [])[:2]:
            events.emit("artifact", stage="papers", role=m["origin"], msg=f"{m['id']}: {p['title']}", url=p["url"])
    write_json(out / "failure_modes.json", modes)
    fresh = sum(m["origin"] == "scan" for m in modes)
    events.emit("stage_end", stage="papers", msg=f"{fresh} paper-backed modes from the scan, {len(modes) - fresh} seeds",
                fresh=fresh, seeds=len(modes) - fresh)
    return modes


MAP = """Task rule (one business rule a refunds job must follow):
  {rule_id}: {sentence}
  Failure mode it is meant to exercise: {mode}

Documented failure modes of AI coding agents (id: description):
{modes}

Which of these documented modes does this rule exercise? A mode counts only when an agent
that has this failure would plausibly get THIS rule wrong.
JSON: {{"supported_by": ["<mode id>", ...], "why": "one sentence"}}"""


def crossref(out: Path, modes: list[dict]) -> dict:
    events.emit("stage_start", stage="crossref", msg="map every task rule to the complaints and papers behind it")
    listing = "\n".join(f"- {m['id']}: {m['mode'][:240]}" for m in modes)
    by_id = {m["id"]: m for m in modes}

    def mapper(rule, agent):
        rid, mode, sentence, _ = rule
        v = llm.ask_json(MAP.format(rule_id=rid, sentence=sentence, mode=mode, modes=listing), CONFIG.judge_model)
        ids = [i for i in v.get("supported_by", []) if i in by_id]
        ev = {"rule": rid, "mode": mode, "supported_by": ids, "why": v.get("why", ""),
              "complaints": [c["url"] for i in ids for c in by_id[i].get("complaints", [])][:8],
              "papers": [p["url"] for i in ids for p in by_id[i].get("papers", [])][:6]}
        ev["_brief"] = f"{rid} <- {', '.join(ids) or 'nothing'}"
        return ev

    found = events.swarm("crossref", "mapper", bizops.RULES, mapper, workers=8, label=lambda r: r[0])
    evidence = {e["rule"]: {k: v for k, v in e.items() if k != "_brief"} for e in found if e}
    write_json(out / "rule_evidence.json", evidence)
    backed = sum(1 for e in evidence.values() if e["papers"])
    events.emit("stage_end", stage="crossref", msg=f"{backed}/{len(bizops.RULES)} rules backed by a paper", backed=backed)
    return evidence


def design(out: Path, evidence: dict, seed: int = 0) -> dict:
    """The impossible task (all rules) and the ladder down: drop the rule least likely to make a solver slip first."""
    events.emit("stage_start", stage="design", msg="stack every rule, then order the ladder down")
    w = mode_weights([{"id": m} for m in {r[1] for r in bizops.RULES}])
    rng = random.Random(seed)
    slip = {rid: rng.betavariate(*w[mode]) + (0.05 if evidence.get(rid, {}).get("papers") else 0.0)
            for rid, mode, _, _ in bizops.RULES}
    order = sorted(bizops.RULE_IDS, key=lambda r: slip[r])  # weakest first: it comes off first
    ladder, rules = [], list(bizops.RULE_IDS)
    for drop in [None] + order[:-1]:
        if drop:
            rules.remove(drop)
        ladder.append({"level": len(rules), "rules": list(rules), "dropped": drop})
        events.emit("work", stage="design", agent="designer-01", role="designer",
                    msg=f"level {len(rules)}: {'all rules' if not drop else 'drop ' + drop}", level=len(rules))
    d = {"family": "bizops-refunds", "slip_prior": {k: round(v, 3) for k, v in slip.items()}, "ladder": ladder,
         "store_id": "S02"}
    write_json(out / "design.json", d)
    events.emit("stage_end", stage="design", msg=f"impossible = {len(bizops.RULE_IDS)} rules; ladder of {len(ladder)} steps")
    return d


def export(out: Path, d: dict, levels: list[int]) -> list[Path]:
    events.emit("stage_start", stage="export", msg=f"Harbor tasks for levels {levels}")
    db = bizops.load_db(out / "business")
    twins = bizops.twin_dbs()
    dirs = []
    for step in d["ladder"]:
        if step["level"] in levels:
            t = bizops.Task(tuple(step["rules"]), {"store_id": d["store_id"]})
            p = bizops.export_harbor(t, db, out / "task", twins)
            n = sum(len(c["tests"]) for c in json.loads((p / "authoring/criteria.json").read_text()))
            events.emit("artifact", stage="export", agent="exporter", role="exporter",
                        msg=f"{p.name}: {len(step['rules'])} criteria, {n} mapped tests", path=str(p))
            dirs.append(p)
    events.emit("stage_end", stage="export", msg=f"{len(dirs)} Harbor tasks")
    return dirs


def run_gate(dirs: list[Path], stages: list[str]) -> dict:
    events.emit("stage_start", stage="gate", msg=f"rsi gate {','.join(stages)}")
    results = {}
    for d in dirs:
        for st in stages:
            r = gate.RUNNERS[st](d)
            events.emit("done" if r["ok"] else "fail", stage="gate", agent=f"gate-{d.name[-11:]}", role="gate",
                        msg=f"{d.name} {st}: {'PASS' if r['ok'] else 'FAIL'} {r.get('note', '')}", gate_stage=st, ok=r["ok"])
            results.setdefault(d.name, {})[st] = r["ok"]
            if not r["ok"]:
                break
    events.emit("stage_end", stage="gate", msg=f"{sum(all(v.values()) for v in results.values())}/{len(results)} tasks passed")
    return results


def run(run_name: str, business: Path = DEFAULT_BUSINESS, skip_scan: bool = False,
        gate_stages: list[str] | None = None, export_levels: list[int] | None = None) -> None:
    out = CONFIG.runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    events.open_log(out, fresh=True)
    events.emit("stage_start", stage="run", msg=f"run {run_name}", models={
        "judge": CONFIG.judge_model, "author": CONFIG.author_model, "solvers": list(CONFIG.solver_models)})
    intake.run(business, out)
    if skip_scan and (out / "complaints.jsonl").exists():
        from .store import read_jsonl
        complaints = read_jsonl(out / "complaints.jsonl")
    else:
        complaints = [] if skip_scan else scan(out)
    modes = read_papers(out, complaints)
    evidence = crossref(out, modes)
    d = design(out, evidence)
    dirs = export(out, d, export_levels or [8, 5, 3])
    if gate_stages:
        write_json(out / "gate.json", run_gate(dirs, gate_stages))
    events.emit("stage_end", stage="run", msg="factory done; Qwen leg runs as its own GPU job (rsi qwen)")
    print(f"swarm: done -> {out}")
