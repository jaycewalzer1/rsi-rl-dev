"""A tiny in-memory ledger."""


class LedgerError(Exception):
    pass


class Ledger:
    def __init__(self, balances: dict[str, int]):
        self.balances = dict(balances)
        self.history: list[tuple[str, str, int]] = []

    def transfer(self, src: str, dst: str, amount: int) -> None:
        self.balances[src] -= amount
        self.balances[dst] += amount
        self.history.append((src, dst, amount))

    def statement(self, account: str) -> list[tuple[str, str, int]]:
        return [h for h in self.history if account in (h[0], h[1])]
