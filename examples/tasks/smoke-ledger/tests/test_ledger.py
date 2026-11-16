import sys

import pytest

sys.path.insert(0, "/app")
from ledger import Ledger, LedgerError  # noqa: E402


def fresh():
    return Ledger({"a": 100, "b": 5})


@pytest.mark.parametrize("amount", [0, -1, 1.5, "3"])
def test_bad_amount_refused(amount):
    led = fresh()
    with pytest.raises(LedgerError):
        led.transfer("a", "b", amount)
    assert led.balances == {"a": 100, "b": 5}
    assert led.statement("a") == []


@pytest.mark.parametrize("src,dst", [("x", "b"), ("a", "x")])
def test_unknown_account_refused(src, dst):
    led = fresh()
    with pytest.raises(LedgerError):
        led.transfer(src, dst, 1)
    assert led.balances == {"a": 100, "b": 5}
    assert led.history == []


def test_overdraft_refused():
    led = fresh()
    with pytest.raises(LedgerError):
        led.transfer("b", "a", 6)
    assert led.balances == {"a": 100, "b": 5}
    assert led.statement("b") == []


def test_valid_transfer_moves_money():
    led = fresh()
    led.transfer("a", "b", 100)
    assert led.balances == {"a": 0, "b": 105}
    assert led.statement("b") == [("a", "b", 100)]
