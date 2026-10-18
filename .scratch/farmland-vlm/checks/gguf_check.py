# Colab CPU: answers from the merged Q4_K_M GGUF (+ merged f16 mmproj) in llama.cpp and mistral.rs.
import subprocess, glob, os, json, time, base64, urllib.request, signal
LBIN = "/content/llamabin/llama-b11719"; MRS = "/content/mrsbin/mistralrs"
Q = "Is there drydown in this tile?"
TILES = sorted(glob.glob("/content/sample/*.jpg"))
out = {}
for tag, tok in [(t, k) for t, k in [("lfm16b", "LiquidAI/LFM2.5-VL-1.6B"), ("qwen2b", "Qwen/Qwen3.5-2B")] if t in os.environ.get("TAGS", "lfm16b,qwen2b")]:
    d = f"/content/merged/{tag}"; r = {}
    for t in TILES:
        o = subprocess.run([f"{LBIN}/llama-mtmd-cli", "-m", f"{d}/model-Q4_K_M.gguf", "--mmproj", f"{d}/mmproj-f16.gguf", "--image", t,
                            "-p", Q, "-n", "12", "--temp", "0", "-t", "2"], capture_output=True, text=True, errors="replace").stdout
        r[os.path.basename(t)] = o.strip()[-60:]
    out[tag] = {"llama_merged_q4km": r}
    p = subprocess.Popen([MRS, "--token-source", "none", "serve", "--port", "8093", "--no-ui", "multimodal", "--cpu", "-m", d,
                          "-f", "model-Q4_K_M.gguf", "--mmproj", f"{d}/mmproj-f16.gguf", "--tok-model-id", tok],
                         stdout=open(f"/content/bench/mrs-{tag}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time(); ok = False
    while time.time() - t0 < 600 and p.poll() is None:
        try: ok = urllib.request.urlopen("http://127.0.0.1:8093/v1/models", timeout=5).status == 200
        except Exception: pass
        if ok: break
        time.sleep(2)
    r = {"load_s": round(time.time() - t0)}
    for t in TILES[:2] if ok else []:
        body = {"model": "default", "temperature": 0, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(t, "rb").read()).decode()}}, {"type": "text", "text": Q}]}]}
        resp = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8093/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
        msg = resp["choices"][0]["message"]; r[os.path.basename(t)] = [msg.get("content"), msg.get("reasoning_content")]
    if not ok: r["error"] = open(f"/content/bench/mrs-{tag}.log").read()[-800:]
    os.killpg(p.pid, signal.SIGTERM)
    out[tag]["mistralrs_merged_q4km"] = r
    print(tag, json.dumps(out[tag]), flush=True)
open("/content/bench/gguf_check2.json", "w").write(json.dumps(out, indent=1))
