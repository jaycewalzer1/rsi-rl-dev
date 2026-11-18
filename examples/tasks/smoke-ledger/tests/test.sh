#!/bin/bash
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
BASE="$(git -C /app rev-list --max-parents=0 HEAD 2> /dev/null | tail -1)"
git -C /app add -A -- . ':(exclude,glob)**/__pycache__/**' > /dev/null 2>&1
git -C /app diff --cached --binary "$BASE" > /logs/verifier/agent.patch 2> /dev/null
cd /tests
if pytest --ctrf /logs/verifier/ctrf.json /tests/test_ledger.py -rA; then
  echo 1 > /logs/verifier/reward.txt
fi
