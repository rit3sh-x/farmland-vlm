#!/bin/bash
# What the Colab run is doing: whether the keepalive is up, its recent log, and the VM's pipeline state, GPU and log tails.
# Usage: scripts/colab/status.sh [refresh seconds]   (with a refresh interval it repeats until Ctrl-C)
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${COLAB_LOG:-$HOME/.cache/farmvlm/colab-keepalive.log}
show() {
  echo "== $(date +%F' '%T)"
  pgrep -f "bash .*keepalive\.sh" >/dev/null && echo "keepalive: running" || echo "keepalive: STOPPED"
  tail -5 "$LOG"
  echo
  timeout 180 colab --auth adc exec -s farmvlm-gpu --timeout 90 -f "$HERE/state.py" 2>&1 | tail -40
}
if [ -z "$1" ]; then show; exit; fi
while :; do clear; show; sleep "$1"; done
