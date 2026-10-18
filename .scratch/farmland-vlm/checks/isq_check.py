# Colab CPU: mistral.rs in-situ quantization (Q4K at load) of the merged Qwen3.5-2B safetensors: load time + peak RSS.
import subprocess, time, urllib.request, json, base64, os, signal, glob
MRS = "/content/mrsbin/mistralrs"; d = "/content/merged/qwen2b"
p = subprocess.Popen([MRS, "--token-source", "none", "serve", "--port", "8094", "--no-ui", "multimodal", "--cpu", "-m", d, "--isq", "q4k"],
                     stdout=open("/content/bench/isq.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
t0 = time.time(); ok = False; peak = 0
def hwm():
    try: return int([l for l in open(f"/proc/{p.pid}/status") if l.startswith("VmHWM")][0].split()[1]) / 1e6
    except Exception: return 0
while time.time() - t0 < 1500 and p.poll() is None:
    try: ok = urllib.request.urlopen("http://127.0.0.1:8094/v1/models", timeout=5).status == 200
    except Exception: pass
    if ok: break
    time.sleep(2)
res = {"isq_load_s": round(time.time() - t0), "peak_rss_gb_after_load": round(hwm(), 2), "ok": ok}
if ok:
    t = sorted(glob.glob("/content/sample/*.jpg"))[0]
    body = {"model": "default", "temperature": 0, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(t, "rb").read()).decode()}}, {"type": "text", "text": "Is there drydown in this tile?"}]}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8094/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    res["answer"] = r["choices"][0]["message"]["content"]; res["peak_rss_gb"] = round(hwm(), 2)
else:
    res["log"] = open("/content/bench/isq.log").read()[-1000:]
os.killpg(p.pid, signal.SIGTERM)
print(json.dumps(res))
