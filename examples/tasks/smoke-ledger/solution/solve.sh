#!/bin/bash
set -e
git -C /app apply --whitespace=nowarn /solution/reference.patch
