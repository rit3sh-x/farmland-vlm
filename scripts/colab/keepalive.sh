#!/bin/bash
# Keep a free Colab T4 working on the pipeline: usage: [DEADLINE=<epoch seconds>] keepalive.sh "<pipeline stages>"
# Checks the session every 7 min; when Colab has recycled it, asks for a new GPU (every 30 min while none is free),
# uploads the mask statistics and relaunches the pipeline, which resumes every run from its last checkpoint on the Hub.
# DEADLINE stops the session and exits before another machine (the weekend Kaggle kernel) takes over the same stages.
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${COLAB_LOG:-$HOME/.cache/farmvlm/colab-keepalive.log}
mkdir -p "$(dirname "$LOG")"
cd "$HERE/../.."
relaunch() {
  colab --auth adc new -s farmvlm-gpu --gpu T4 2>&1 | tail -1 | grep -q READY || { echo "$(date +%F' '%T) no GPU" >> "$LOG"; return 1; }
  colab --auth adc upload -s farmvlm-gpu data/stats.npz /content/stats.npz >/dev/null 2>&1
  colab --auth adc exec -s farmvlm-gpu --timeout 120 -f "$HERE/relaunch.py" --env "HF_TOKEN=$(cat ~/.config/farmvlm/hf_token)" --env "STAGES=$1" 2>&1 | tail -1 >> "$LOG"
  echo "$(date +%F' '%T) relaunched: $1" >> "$LOG"
}
for i in $(seq 1 300); do
  if [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
    colab --auth adc stop -s farmvlm-gpu >/dev/null 2>&1; echo "$(date +%F' '%T) deadline: session stopped" >> "$LOG"; exit 0
  fi
  o=$(colab --auth adc exec -s farmvlm-gpu --timeout 90 -f "$HERE/state.py" 2>&1)
  if echo "$o" | grep -q -E "not found|No active"; then relaunch "$1" || { sleep 1800; continue; }
  elif echo "$o" | grep -q PIPELINE_EXITED; then echo "$(date +%F' '%T) pipeline exited" >> "$LOG"; echo "$o" | tail -40; exit 0
  fi
  sleep 420
done
