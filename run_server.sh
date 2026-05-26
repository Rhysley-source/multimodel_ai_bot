#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
"$SCRIPT_DIR/venv/bin/python" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
