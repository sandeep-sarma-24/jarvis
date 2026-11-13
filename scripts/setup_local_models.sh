#!/usr/bin/env bash
# Pull the configured reasoning model into the local Ollama, only if it is absent.
# Idempotent: does nothing when the model is installed or a pull of it is already running.
# Never pulls any other model. Usage: scripts/setup_local_models.sh
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
read -r model url < <("$PY" -c 'import tomllib
r = tomllib.load(open("configs/default.toml", "rb"))["reasoning"]
print(r["model"], r["ollama_url"])')
url=${url%/}

if ! "$PY" -c 'import sys; from app.config import is_loopback; sys.exit(not is_loopback(sys.argv[1]))' "$url"; then
    echo "FAIL  reasoning.ollama_url=$url is not loopback; refusing (JARVIS runs locally only)"; exit 1
fi
if ! curl -sf -m 3 "$url/api/version" >/dev/null; then
    echo "FAIL  Ollama is not answering at $url"
    echo "      start it with: ollama serve   (or open /Applications/Ollama.app)"; exit 1
fi
echo "PASS  Ollama $(curl -s -m 3 "$url/api/version") at $url"

installed() { ollama list | awk 'NR>1 {print $1}' | grep -qx "$model"; }
if installed; then
    echo "PASS  $model already installed; not pulling"
elif pgrep -f "ollama pull $model" >/dev/null; then
    echo "WAIT  a pull of $model is already running; not starting another. Re-run when it finishes."; exit 2
else
    echo "....  pulling $model (only this model)"
    OLLAMA_HOST="$url" ollama pull "$model"
fi

installed || { echo "FAIL  $model not listed after pull"; exit 1; }
echo "PASS  $model listed:"
ollama list | awk -v m="$model" 'NR==1 || $1==m' | sed 's/^/        /'
echo "      ~/.ollama/models total: $(du -sh ~/.ollama/models 2>/dev/null | cut -f1)"
