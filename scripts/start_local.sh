#!/usr/bin/env bash
# One command: make sure local Ollama and the model are up, prove Gemma answers, then start JARVIS.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null; then
  echo "Starting Ollama…"
  nohup ollama serve >/tmp/jarvis-ollama.log 2>&1 &
  for _ in $(seq 20); do curl -s -m 1 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 0.5; done
fi
scripts/setup_local_models.sh
.venv/bin/python -m scripts.verify_local_gemma
echo
# Models are already cached; without this the ASR loader asks huggingface.co for the latest revision.
export HF_HUB_OFFLINE=1
echo "JARVIS is starting. When you see it listening, say: \"Jarvis, what is the capital of France?\"  (Ctrl+C to quit)"
exec .venv/bin/python -m app.main
