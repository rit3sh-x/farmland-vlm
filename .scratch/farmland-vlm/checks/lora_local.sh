#!/bin/bash
# Fine-tuned behaviour on the laptop: official base GGUF + small adapter from Colab, both runtimes.
S=/tmp/claude-1000/-home-rit3sh-project-ai512/7b5c81d6-ca8f-42f6-bc78-a5cbbf7b1cb8/scratchpad
L=$S/bin/llamacpp/llama-b11719; B=$S/bin/mistralrs/mistralrs; M=$S/models/$1; A=$S/adapters/$1
export TRAIN_PROMPTS=1
python3 -I $S/bench_serve.py llama-$1-base 8091 $S/tiles -- $L/llama-server -m $M/$2 --mmproj $M/$3 -t 12 -c 4096 --port 8091 --no-webui > $S/logs/lora-llama-$1-base.json
python3 -I $S/bench_serve.py llama-$1-lora 8091 $S/tiles -- $L/llama-server -m $M/$2 --mmproj $M/$3 --lora $A/lora-f16.gguf -t 12 -c 4096 --port 8091 --no-webui > $S/logs/lora-llama-$1-lora.json
ADAPTER=ft python3 -I $S/bench_serve.py mrs-$1-lora 8092 $S/tiles -- $B --token-source none serve --port 8092 --no-ui multimodal --cpu -m $M -f $2 --mmproj $M/$3 --tok-model-id $4 --lora ft=$A > $S/logs/lora-mrs-$1-lora.json
python3 -I $S/bench_serve.py mrs-$1-base 8092 $S/tiles -- $B --token-source none serve --port 8092 --no-ui multimodal --cpu -m $M -f $2 --mmproj $M/$3 --tok-model-id $4 --lora ft=$A > $S/logs/lora-mrs-$1-base.json
for f in $S/logs/lora-*-$1-*.json; do python3 -I -c '
import json,sys
p=sys.argv[1]
try: d=json.load(open(p))
except Exception: print(p, open(p).read()[:1500]); sys.exit()
if "error" in d: print(d); sys.exit()
print(d["label"], "| answer wall", d["answer"]["median_wall_s"], "| caption wall", d["caption"]["median_wall_s"])
print("   answers:", d["samples"]["answer"]); print("   captions:", d["samples"]["caption"])' $f; done
