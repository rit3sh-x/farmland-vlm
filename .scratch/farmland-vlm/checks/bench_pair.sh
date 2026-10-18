#!/bin/bash
# bench_pair.sh <tag> <dir> <model.gguf> <mmproj.gguf> <tok-model-id>: llama-server then mistralrs on the same GGUF
S=/tmp/claude-1000/-home-rit3sh-project-ai512/7b5c81d6-ca8f-42f6-bc78-a5cbbf7b1cb8/scratchpad
L=$S/bin/llamacpp/llama-b11719; B=$S/bin/mistralrs/mistralrs
F='^\s+(\]|\[|-?[0-9.e-]+,?|"bytes".*|\{|\},?|"id".*|"token".*|"logprob".*)$'
python3 -I $S/bench_serve.py llama-$1 8091 $S/tiles -- $L/llama-server -m $2/$3 --mmproj $2/$4 -t 16 -c 4096 --port 8091 --no-webui > $S/logs/res-llama-$1.json
python3 -I $S/bench_serve.py mrs-$1 8092 $S/tiles -- $B --token-source none serve --port 8092 --no-ui multimodal --cpu -m $2 -f $3 --mmproj $2/$4 --tok-model-id $5 > $S/logs/res-mrs-$1.json
