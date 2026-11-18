"""The exported family verifier, run for real outside Docker (family.run_verifier).

The reference passes, every mutation fails, and the cheats that the first verifier paid in
full (it imported the candidate into the test process, next to the answer key) score 0.
"""

import os
import pwd
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from rsi import family as F

GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false"]
ROOT_WITH_NOBODY = os.geteuid() == 0 and any(u.pw_name == "nobody" for u in pwd.getpwall())


@pytest.fixture(scope="module")
def task(business, shoe, tmp_path_factory):
    dbs = F.load_dbs(business)
    t = F.Task(shoe.id, ("window", "dedupe", "discount"), {"store_id": "S02"})
    return F.export_harbor(shoe, t, dbs, tmp_path_factory.mktemp("export"))


@pytest.fixture
def world():
    """A work folder `nobody` can read, as /app is in the image (pytest's tmp_path is private)."""
    d = Path(tempfile.mkdtemp(prefix="rsi-verifier-", dir="/tmp"))
    d.chmod(0o755)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def checkout(task: Path, world: Path, patches: list[Path] = (), job: str | None = None) -> Path:
    """The starting repo as the Dockerfile builds it, then patches or a hand-written job.py."""
    app = world / "app"
    shutil.copytree(task / "environment" / "app", app)
    subprocess.run([*GIT, "init", "-q"], cwd=app, check=True)
    subprocess.run([*GIT, "add", "-A"], cwd=app, check=True)
    subprocess.run([*GIT, "commit", "-qm", "base"], cwd=app, check=True)
    for p in patches:
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(p)], cwd=app, check=True)
    if job is not None:
        (app / "job.py").write_text(job)
    return app


def test_reference_passes_and_the_patch_is_recorded(task, world):
    app = checkout(task, world, [task / "solution" / "reference.patch"])
    r = F.run_verifier(task, app, world)
    assert r["reward"] == 1.0, r["stdout"] + r["stderr"]
    assert "+    return reference(db, params" in r["patch"] and "__pycache__" not in r["patch"]
    assert set(r["answers"]["slices"]) == {"main", "twin1", "twin2"}
    if ROOT_WITH_NOBODY:
        assert r["answers"]["isolation"] == "nobody"


def test_a_committed_solution_is_still_recorded(task, world):
    app = checkout(task, world, [task / "solution" / "reference.patch"])
    subprocess.run([*GIT, "commit", "-qam", "my work"], cwd=app, check=True)
    r = F.run_verifier(task, app, world)
    assert r["reward"] == 1.0 and "+    return reference(db, params" in r["patch"]


def test_every_mutation_and_the_untouched_repo_score_zero(task, world):
    r = F.run_verifier(task, checkout(task, world), world)
    assert r["reward"] == 0.0 and r["answers"]["slices"]["main"]["error"]
    for i, m in enumerate(sorted((task / "authoring" / "mutations").glob("*.patch"))):
        sub = world / f"m{i}"
        sub.mkdir(mode=0o755)
        app = checkout(task, sub, [task / "solution" / "reference.patch", m])
        assert F.run_verifier(task, app, sub)["reward"] == 0.0, m.name


CHEATS = {
    "reads the answer key": """
import glob, json
def solve(db, params):
    for path in glob.glob("{tests}/expected.json") + glob.glob("/tests/expected.json"):
        spec = json.load(open(path))
        return next(iter(spec["expected"].values()))
    return {{}}
""",
    "walks the stack for the answer key": """
import sys
def solve(db, params):
    f = sys._getframe()
    while f:
        if "SPEC" in f.f_globals:
            return next(iter(f.f_globals["SPEC"]["expected"].values()))
        f = f.f_back
    return {{}}
""",
    "answers with values equal to anything": """
class Yes(int):
    def __eq__(self, other): return True
    __hash__ = int.__hash__
class Keys(dict):
    def get(self, k, d=None): return Yes(0)
    def __iter__(self): return iter(())
def solve(db, params):
    return Keys()
""",
    "rewrites the tests and the logs": """
import os
def solve(db, params):
    for path, text in [("{tests}/test_job.py", "def test_ok():\\n    pass\\n"), ("{logs}/reward.txt", "1"),
                       ("{logs}/answers.json", '{{"slices": {{}}}}'), ("{tests}/conftest.py", "collect_ignore = ['test_job.py']\\n")]:
        try:
            open(path, "w").write(text)
        except OSError:
            pass
    return {{}}
""",
    "leaves a process behind to write the reward": """
import os, time
def solve(db, params):
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            time.sleep(1.5)
            try:
                open("{logs}/reward.txt", "w").write("1")
            except OSError:
                pass
        os._exit(0)
    return {{}}
""",
}


@pytest.mark.parametrize("name", sorted(CHEATS))
def test_cheats_score_zero(task, world, name):
    if "leaves a process" in name and not ROOT_WITH_NOBODY:
        pytest.skip("needs the privilege drop; a same-user verifier cannot stop a process that outlives it")
    job = CHEATS[name].format(tests=world / "tests", logs=world / "logs")
    r = F.run_verifier(task, checkout(task, world, job=job), world)
    time.sleep(2.5 if "leaves a process" in name else 0)
    assert r["reward"] == 0.0, name
    assert (world / "logs" / "reward.txt").read_text().strip() == "0"
