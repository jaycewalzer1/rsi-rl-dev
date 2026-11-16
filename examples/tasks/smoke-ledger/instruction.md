Harden `Ledger.transfer` in `/app/ledger.py`.

- A transfer whose amount is not a positive integer is refused.
- A transfer from or to an account that does not exist is refused.
- A transfer that would take the source balance below zero is refused.
- A refused transfer changes nothing: no balance moves and nothing is added to any statement.

Pins: every refusal raises `LedgerError`.
