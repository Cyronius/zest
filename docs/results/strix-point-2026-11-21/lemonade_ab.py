# Python port of start-lemonade.ps1 + bench-lemonade.ps1 (2026-11-21): same settings, no PowerShell.
import json, os, statistics, subprocess, sys, time, urllib.request, urllib.error, socket

lemond, llama_dir, state, port, out, model_dir = sys.argv[1:7]
models = sys.argv[7:]
rounds, max_tokens = int(os.environ.get("AB_ROUNDS", "3")), 16
copies = int(os.environ.get("AB_COPIES", "8"))
os.makedirs(f"{state}/cache", exist_ok=True); os.makedirs(f"{state}/config", exist_ok=True)
base = f"http://127.0.0.1:{port}/api/v1"

cfg = {"config_version": 2, "host": "127.0.0.1", "port": int(port), "broadcast": False, "offline": True,
       "no_fetch_executables": True, "log_level": "debug", "log_file": f"{state}/server.log",
       "models_dir": f"{state}/models", "extra_models_dir": model_dir,
       "llamacpp": {"backend": "vulkan", "vulkan_bin": f"{llama_dir}/llama-server.exe"}}
json.dump(cfg, open(f"{state}/config/config.json", "w"), indent=1)
json.dump({}, open(f"{state}/config/recipe_options.json", "w"))
env = dict(os.environ, GGML_BACKEND_PATH=f"{llama_dir}/ggml-zest.dll", GGML_XDNA_TRACE="1")
env.pop("GGML_XDNA_HOST_ONLY", None)
srv = subprocess.Popen([lemond, f"{state}/cache", f"{state}/config", "--port", port, "--host", "127.0.0.1",
                        "--no-broadcast"], env=env, stdout=open(f"{state}/lemond.out", "w"), stderr=subprocess.STDOUT)


def post(route, body):
    req = urllib.request.Request(f"{base}/{route}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        if route == "unload" and e.code == 404:
            return {}
        raise RuntimeError(f"{route}: {e.code} {e.read()[:500]}")


for _ in range(120):
    try:
        urllib.request.urlopen(f"{base}/health", timeout=2); break
    except Exception:
        time.sleep(1)
else:
    srv.kill(); sys.exit("server did not start")

passage = open(os.environ.get("AB_TEXT") or os.path.join(os.path.dirname(__file__), "prose.txt"), encoding="utf-8-sig", newline="").read()
prompt = "\n".join(f"Passage {i}:\n{passage}" for i in range(1, copies + 1)) + "\nSummarize the passage in one sentence. /no_think"
summary = {}
try:
    for mfile in models:
        mid = "extra." + os.path.splitext(mfile)[0]
        req = {"model": mid, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
               "temperature": 0, "stream": False, "cache_prompt": False,
               "chat_template_kwargs": {"enable_thinking": False}}
        rows = []
        for rnd in range(1, rounds + 1):
            for mode in (("gpu", "npu") if rnd % 2 else ("npu", "gpu")):
                post("unload", {"model_name": mid})
                dev, split = ("XDNA0,Vulkan0", "0,1") if mode == "npu" else ("Vulkan0", "1")
                post("load", {"model_name": mid, "ctx_size": 8192, "llamacpp_backend": "vulkan", "llamacpp_device": dev,
                              "llamacpp_args": f"-ts {split} -ub 2048 -b 2048 --flash-attn on --parallel 1",
                              "merge_args": False})
                post("chat/completions", req)
                r = post("chat/completions", req)
                t = r["timings"]
                assert t["prompt_n"] >= 2048 and t.get("cache_n", 0) == 0, t
                rows.append({"round": rnd, "mode": mode, "n": t["prompt_n"], "ms": t["prompt_ms"], "tps": t["prompt_per_second"]})
                print(f"{mfile} round {rnd} {mode}: {t['prompt_n']} tok, {t['prompt_per_second']:.1f} tok/s", flush=True)
        post("unload", {"model_name": mid})
        ratios = []
        for rnd in range(1, rounds + 1):
            g = next(x for x in rows if x["round"] == rnd and x["mode"] == "gpu")
            n = next(x for x in rows if x["round"] == rnd and x["mode"] == "npu")
            assert g["n"] == n["n"]
            ratios.append(g["ms"] / n["ms"])
        summary[mfile] = {"prompt_tokens": rows[0]["n"], "rows": rows,
                          "gpu_median": statistics.median(x["tps"] for x in rows if x["mode"] == "gpu"),
                          "npu_median": statistics.median(x["tps"] for x in rows if x["mode"] == "npu"),
                          "ratio_median": statistics.median(ratios), "ratio_lo": min(ratios), "ratio_hi": max(ratios)}
        s = summary[mfile]
        print(f"== {mfile}: GPU {s['gpu_median']:.0f}  NPU {s['npu_median']:.0f} tok/s  {s['ratio_median']:.2f}x ({s['ratio_lo']:.2f}-{s['ratio_hi']:.2f})", flush=True)
        if os.environ.get("AB_CHECKS"):
            q = subprocess.run([sys.executable, "-I", os.environ["AB_CHECKS"], "--model", mid, "--base-url", base,
                                "--mode", "both", "--max-tokens", "256", "--output", out + "." + mid + ".quality.json"],
                               capture_output=True, text=True)
            print(q.stdout, flush=True)
            checks = json.load(open(out + "." + mid + ".quality.json"))["checks"]
            s["answers"] = {m: sum(c["answer_correct"] for c in checks if c["mode"] == m) for m in ("gpu", "npu")}
            print(f"== {mfile}: answers GPU {s['answers']['gpu']}/3  NPU {s['answers']['npu']}/3", flush=True)
        json.dump(summary, open(out, "w"), indent=1)
finally:
    srv.kill()
