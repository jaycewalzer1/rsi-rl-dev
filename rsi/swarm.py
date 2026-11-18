"""The whole factory as a swarm of agents, logged to runs/<run>/events.jsonl for the demo page.

    intake     loader per table, pruner per table, slicer, steward agent per table (judge model)
    scan       HN scouts (Algolia API, one per query) + X scouts (X API, or web-search snippets)
               then labeller agents, 25 posts each, keep concrete failures only
    papers     one reader agent per complaint category: arXiv + Semantic Scholar, judge keeps
               the category only if a paper measures it; seed modes merged in
    author     planner + one author agent per task family (author model) write the families for
               THIS dataset; the mechanical checker and a cold solve by a different model decide
    export     each accepted family's top level (every rule on) as a Harbor task
    gate       your pipeline: `rsi gate` shape, oracle, nop, mutations on the exported tasks
    qwen       (separate GPU-queue job, `rsi qwen`) calibrate, benchmark, post-train, benchmark

    rsi factory --run NAME --business DIR [--modes-from RUN] [--families 3] [--gate STAGES]
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import author, events, family as F, gate, intake, papers, scan_hn, scan_x
from .config import CONFIG
from .store import read_json, read_jsonl, write_json, write_jsonl

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
    write_json(out / "failure_modes.json", modes)
    fresh = sum(m["origin"] == "scan" for m in modes)
    events.emit("stage_end", stage="papers", msg=f"{fresh} paper-backed modes from the scan, {len(modes) - fresh} seeds",
                fresh=fresh, seeds=len(modes) - fresh)
    return modes


def reuse_modes(out: Path, other: Path) -> list[dict]:
    """Carry the scan and paper check over from an earlier run instead of repeating them."""
    for name in ("complaints.jsonl", "failure_modes.json"):
        shutil.copy(other / name, out / name)
    complaints, modes = read_jsonl(out / "complaints.jsonl"), read_json(out / "failure_modes.json")
    events.emit("stage_start", stage="scan", msg=f"reusing the scan and paper check from run {other.name}")
    for c in complaints:
        events.emit("artifact", stage="scan", role=c["provenance"], msg=c["failure"], url=c["url"], category=c["category"])
    events.emit("stage_end", stage="scan", msg=f"{len(complaints)} concrete failures, reused from run {other.name}")
    events.emit("stage_start", stage="papers", msg=f"reused from run {other.name}")
    events.emit("stage_end", stage="papers", msg=f"{len(modes)} failure modes ({sum(m['origin'] == 'scan' for m in modes)} paper-backed from the scan)")
    return modes


def emit_evidence(modes: list[dict]) -> None:
    for m in modes:
        for p in m.get("papers", [])[:2]:
            events.emit("artifact", stage="papers", role=m["origin"], msg=f"{m['id']}: {p['title']}", url=p["url"])


def export(out: Path) -> list[Path]:
    """Each accepted family's hardest task, with params under which every rule changes the answer."""
    events.emit("stage_start", stage="export", msg="top level of every accepted family as a Harbor task")
    dbs = F.load_dbs(out / "business")
    dirs = []
    for fam in F.load_run(out):
        full = tuple(fam.rule_ids)
        best = max(fam.params_choices(dbs[0]), key=lambda p: min(F.effect(fam, dbs, p, full).values()))
        d = F.export_harbor(fam, F.Task(fam.id, full, best), dbs, out / "task")
        n = sum(len(c["tests"]) for c in read_json(d / "authoring" / "criteria.json"))
        events.emit("artifact", stage="export", agent="exporter", role="exporter",
                    msg=f"{d.name}: {len(full)} criteria, {n} mapped tests", path=str(d))
        dirs.append(d)
    events.emit("stage_end", stage="export", msg=f"{len(dirs)} Harbor tasks")
    return dirs


def run_gate(dirs: list[Path], stages: list[str]) -> dict:
    events.emit("stage_start", stage="gate", msg=f"rsi gate {','.join(stages)}")
    results = {}
    for d in dirs:
        for st in stages:
            r = gate.RUNNERS[st](d)
            events.emit("done" if r["ok"] else "fail", stage="gate", agent=f"gate-{d.name[:28]}", role="gate",
                        msg=f"{d.name} {st}: {'PASS' if r['ok'] else 'FAIL'} {r.get('note', '')}", gate_stage=st, ok=r["ok"])
            results.setdefault(d.name, {})[st] = r["ok"]
            if not r["ok"]:
                break
    events.emit("stage_end", stage="gate", msg=f"{sum(all(v.values()) for v in results.values())}/{len(results)} tasks passed")
    return results


def run(run_name: str, business: Path = DEFAULT_BUSINESS, *, modes_from: str | None = None, families: int = 3,
        gate_stages: list[str] | None = None, rows: int = 1500, only: list[str] | None = None) -> None:
    """`only` limits the run to some stages (intake, modes, author, export, gate) and keeps the log."""
    out = CONFIG.runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    todo = only or ["intake", "modes", "author", "export", "gate"]
    events.open_log(out, fresh=only is None)
    if only is None:
        events.emit("stage_start", stage="run", msg=f"run {run_name}", models={
            "judge": CONFIG.judge_model, "author": CONFIG.author_model, "check": CONFIG.check_model})
    if "intake" in todo:
        intake.run(business, out, rows=rows)
    if "modes" in todo:
        modes = reuse_modes(out, CONFIG.runs_dir / modes_from) if modes_from else read_papers(out, scan(out))
        emit_evidence(modes)
    if "author" in todo:
        author.run(out, families)
    if "export" in todo or "gate" in todo:
        dirs = export(out)
        if "gate" in todo and gate_stages:
            write_json(out / "gate.json", run_gate(dirs, gate_stages))
    if only is None:
        events.emit("stage_end", stage="run", msg="factory done; the Qwen leg runs as its own GPU job (rsi qwen)")
    print(f"factory: done -> {out}")
