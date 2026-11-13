#!/usr/bin/env bash
# Read-only check that this Mac can run the whole JARVIS stack locally.
# Prints PASS/WARN/FAIL lines; exits 1 if any hard requirement FAILs.
# Usage: scripts/check_local_environment.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
CONFIG=configs/default.toml
fails=0
pass() { echo "PASS  $*"; }
warn() { echo "WARN  $*"; }
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }
info() { echo "      $*"; }

cfg() {  # cfg section.key -> value from the config, via tomllib
    "$PY" - "$CONFIG" "$1" <<'EOF'
import sys, tomllib
v = tomllib.load(open(sys.argv[1], "rb"))
for k in sys.argv[2].split("."):
    v = v.get(k, "") if isinstance(v, dict) else ""
print(v)
EOF
}

# Platform
[ "$(uname -s)" = Darwin ] && pass "macOS $(sw_vers -productVersion)" || fail "not macOS ($(uname -s))"
[ "$(uname -m)" = arm64 ] && pass "Apple Silicon: $(sysctl -n machdep.cpu.brand_string)" \
    || fail "not arm64 ($(uname -m)); MLX needs Apple Silicon"
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$mem_gb" -ge 16 ] && pass "memory ${mem_gb} GB" || warn "memory ${mem_gb} GB (stack was sized for 16 GB)"

# Python
if [ -x "$PY" ]; then
    pyver=$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
    "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' && pass "venv python $pyver" \
        || fail "venv python $pyver < 3.11"
else
    fail "no $PY; create the venv first"
    echo "RESULT: $fails hard requirement(s) failed"; exit 1
fi

# Python packages
for pkg in mlx mlx-audio ptts onnxruntime torch silero-vad sounddevice httpx numpy psutil; do
    v=$("$PY" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null)
    case "$pkg" in
        psutil) [ -n "$v" ] && pass "$pkg $v" || warn "$pkg missing (benchmarks need it)" ;;
        *) [ -n "$v" ] && pass "$pkg $v" || fail "$pkg missing in venv" ;;
    esac
done

# Config endpoints must be loopback
model=$(cfg reasoning.model)
url=$(cfg reasoning.ollama_url)
loop=$("$PY" -c 'from app.config import load_config, is_loopback
c = load_config()
urls = {"reasoning.ollama_url": c.reasoning.ollama_url, "perception.vision.ollama_url": c.perception.vision.ollama_url}
if c.reasoning.fallback_model: urls["reasoning.fallback_url"] = c.reasoning.fallback_url
bad = [f"{k}={v}" for k, v in urls.items() if not is_loopback(v)]
print("local_only=%s %s" % (c.runtime.local_only, "OK" if not bad else "NON-LOOPBACK " + " ".join(bad)))' 2>&1 | tail -1)
case "$loop" in
    *" OK") pass "model URLs are loopback ($loop)" ;;
    *) fail "model URLs: $loop" ;;
esac

# Ollama
if command -v ollama >/dev/null; then
    pass "ollama $(ollama --version 2>&1 | sed -n 's/.*version is //p' | head -1) at $(command -v ollama)"
else
    fail "ollama binary not on PATH (brew install ollama)"
fi
if curl -sf -m 3 "$url/api/version" >/dev/null; then
    pass "Ollama server answers at $url ($(curl -s -m 3 "$url/api/version"))"
    echo "      installed models:"
    ollama list | sed 's/^/        /'
    if ollama list | awk 'NR>1 {print $1}' | grep -qx "$model"; then
        pass "configured reasoning.model $model is installed"
    elif pgrep -f "ollama pull $model" >/dev/null; then
        fail "reasoning.model $model not installed yet (a pull is in progress)"
    else
        fail "reasoning.model $model not installed (scripts/setup_local_models.sh)"
    fi
    echo "      loaded models (ollama ps):"
    ollama ps | sed 's/^/        /'
else
    fail "Ollama server not answering at $url (start it: ollama serve, or open Ollama.app)"
fi

# Resources
free_gb=$(df -g . | awk 'NR==2 {print $4}')
[ "$free_gb" -ge 10 ] && pass "free disk ${free_gb} GB" || warn "free disk ${free_gb} GB (< 10 GB)"
mp=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
if [ -z "$mp" ]; then warn "memory_pressure unavailable"
elif [ "$mp" -ge 30 ]; then pass "memory free ${mp}%"
else warn "memory free ${mp}% (pressure is high; close apps before benchmarking)"; fi
info "swap: $(sysctl -n vm.swapusage)"
therm=$(pmset -g therm 2>/dev/null)
if echo "$therm" | grep -q "No thermal warning level has been recorded" \
   || ! echo "$therm" | grep -q "level"; then
    pass "thermal: no thermal warning recorded"
else
    warn "thermal: $(echo "$therm" | tr '\n' ' ')"
fi

echo
if [ "$fails" -gt 0 ]; then echo "RESULT: $fails hard requirement(s) failed"; exit 1; fi
echo "RESULT: all hard requirements passed"
