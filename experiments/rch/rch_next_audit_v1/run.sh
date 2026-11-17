#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python -u experiments/rch_next_audit_v1/run_next_audit.py "$@"
