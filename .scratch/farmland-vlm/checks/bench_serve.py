# Start an OpenAI-compatible server (llama-server or mistralrs), time load + caption/answer requests on tiles.
# usage: bench_serve.py <label> <port> <tiles_dir> -- <server cmd...>
import sys, os, time, json, base64, subprocess, urllib.request, glob, signal
label, port, tiles = sys.argv[1], sys.argv[2], sys.argv[3]
cmd = sys.argv[sys.argv.index("--") + 1:]
llama = "llama-server" in cmd[0]
logf = f"/tmp/claude-1000/-home-rit3sh-project-ai512/7b5c81d6-ca8f-42f6-bc78-a5cbbf7b1cb8/scratchpad/logs/{label}.log"
os.makedirs(os.path.dirname(logf), exist_ok=True)
base = f"http://127.0.0.1:{port}"
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT, start_new_session=True)
def hwm():
    try: return int([l for l in open(f"/proc/{proc.pid}/status") if l.startswith("VmHWM")][0].split()[1]) / 1e6
    except Exception: return None
def get(path):
    return urllib.request.urlopen(base + path, timeout=5).status
ready = False
while time.time() - t0 < 1800 and proc.poll() is None:
    try:
        if get("/health") == 200 and get("/v1/models") == 200: ready = True; break
    except Exception: pass
    time.sleep(0.5)
load_s = time.time() - t0
if not ready:
    print(json.dumps({"label": label, "error": "server not ready", "rc": proc.poll(), "load_s": round(load_s)}))
    print("".join(open(logf).readlines()[-25:])); sys.exit(1)
mid = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
PROMPTS = {"caption": ("Describe the crop field anomalies visible in this aerial image in one sentence.", 60),
           "answer": ("Is there drydown in this image? Answer yes or no.", 20)}
if os.environ.get("TRAIN_PROMPTS"):  # the wording the LoRA adapters were trained on
    PROMPTS = {"caption": ("Describe the anomalies in this aerial farmland tile.", 60), "answer": ("Is there drydown in this tile?", 20)}
def ask(img, task):
    text, n = PROMPTS[task]
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    body = {"model": mid, "temperature": 0, "max_tokens": n, "logprobs": True, "top_logprobs": 3,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                                                      {"type": "text", "text": text}]}]}
    if llama: body.update(cache_prompt=False, ignore_eos=not os.environ.get("TRAIN_PROMPTS"))
    if os.environ.get("ADAPTER"): body["adapter"] = os.environ["ADAPTER"]
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    wall = time.time() - t
    ch = r["choices"][0]; u = r.get("usage", {})
    lp = ch.get("logprobs") or {}
    return {"wall": wall, "prompt_tok": u.get("prompt_tokens"), "gen_tok": u.get("completion_tokens"),
            "timings": r.get("timings"), "usage": u, "text": ch["message"].get("content"),
            "logprobs": bool(lp.get("content")), "lp0": (lp.get("content") or [{}])[0].get("top_logprobs")}
imgs = sorted(glob.glob(tiles + "/*.jpg"))
try:
    ask(imgs[-1], "answer")  # warmup
    res = {k: [ask(i, k) for i in imgs] for k in PROMPTS}
except Exception as e:
    print(json.dumps({"label": label, "error": repr(e)[:300], "load_s": round(load_s, 1), "peak_rss_gb": hwm()}))
    print("".join(open(logf).readlines()[-25:])); sys.exit(1)
peak = hwm()
os.killpg(proc.pid, signal.SIGTERM)
out = {"label": label, "load_s": round(load_s, 1), "peak_rss_gb": round(peak, 2)}
for k, rs in res.items():
    walls = sorted(x["wall"] for x in rs)
    out[k] = {"median_wall_s": round(walls[len(walls) // 2], 2), "max_wall_s": round(walls[-1], 2),
              "prompt_tok": rs[0]["prompt_tok"], "gen_tok": [x["gen_tok"] for x in rs]}
    t = rs[0]["timings"] or {}
    if t: out[k]["prefill_ms"] = round(t.get("prompt_ms", 0)); out[k]["decode_tok_s"] = round(t.get("predicted_per_second", 0), 1)
    u = rs[0]["usage"]
    if "avg_compl_tok_per_sec" in u or "completion_tok_per_sec" in u:
        out[k]["usage_extra"] = {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in u.items() if kk not in ("prompt_tokens", "completion_tokens", "total_tokens")}
out["logprobs"] = all(x["logprobs"] for rs in res.values() for x in rs)
out["lp_example"] = res["answer"][0]["lp0"]
out["samples"] = {k: {os.path.basename(x): rs[j]["text"] for j, x in enumerate(imgs)} for k, rs in res.items()}
print(json.dumps(out, indent=1))
