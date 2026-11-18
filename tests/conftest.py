import pytest

from rsi import business_gen, family as F, intake


@pytest.fixture(scope="session")
def business(tmp_path_factory):
    """The synthetic shoe store, pruned and cut into three slices."""
    out = tmp_path_factory.mktemp("run")
    src = out / "raw"
    business_gen.write_csv(business_gen.generate(7), src)
    return intake.run(src, out, stewards=False, rows=800)


@pytest.fixture(scope="session")
def shoe():
    return F.load(F.BUILTIN / "shoe_store.py")
