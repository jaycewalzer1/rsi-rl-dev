#!/bin/bash
mkdir -p /logs/verifier
cd /tests
if pytest --ctrf /logs/verifier/ctrf.json /tests/test_ledger.py -rA; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
