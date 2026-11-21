#!/usr/bin/env python3
"""bench.ps1's measurement for machines without PowerShell: how much faster llama.cpp reads a prompt with the NPU
add-on than on the GPU alone, and optionally how far its answers drift, for one or more models.

    tools/bench.py ~/models/Qwen3-4B-Q4_K_M.gguf
    tools/bench.py --rounds 9 --kl --text docs.txt a.gguf b.gguf

Each round is one llama-bench process that tests both the GPU alone and the GPU with the add-on, so the model loads
once and the two measurements sit next to each other in time. A round's ratio compares that pair; the table gives
the median tokens a second over the rounds and, in brackets, the slowest and fastest round's ratio to the GPU alone.

Results vary from run to run: the processor, GPU and NPU share one chip, its memory and its power budget. Close
other programs first. On models under about 3B parameters a round takes seconds, so give those more rounds
(--rounds 9).

--no-split adds a second measurement per round with the row split off (GGML_XDNA_SPLIT=0: the NPU takes whole
pieces and the GPU waits its turn), in a process of its own with its own GPU-alone figure.

--kl also runs llama-perplexity on the GPU alone and then with the add-on over the same text (--text), and reports
the mean KL divergence between the two and how often both pick the same most likely next token.
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description="Prompt reading speed with the NPU add-on against the GPU alone")
ap.add_argument("models", nargs="+", metavar="model.gguf")
ap.add_argument("--rounds", type=int, default=5, help="rounds per model; each round runs both devices (default 5)")
ap.add_argument("--prompt", type=int, default=2048, help="prompt length in tokens, read in 2,048-token chunks as zest does")
ap.add_argument("--no-split", action="store_true", help="also measure with the row split off")
ap.add_argument("--kl", action="store_true", help="also measure answer drift with llama-perplexity (needs --text)")
ap.add_argument("--text", help="the text for --kl: plain text, at least chunks x 512 tokens")
ap.add_argument("--chunks", type=int, default=8, help="512-token pieces of --text to compare; 8 is about 4,000 tokens")
ap.add_argument("--llama-dir", help="the folder with zest and llama.cpp's programs")
ap.add_argument("--out", help="also write every round's figures to this JSON file")
a = ap.parse_args()

# next to this script (the release package), else the repo's build of llama.cpp
llama_dir = a.llama_dir or (HERE if os.path.exists(os.path.join(HERE, "zest")) else os.path.join(HERE, "..", "third_party", "llama-b10944"))
zest = os.path.join(llama_dir, "zest")
if not os.path.exists(zest):
    sys.exit(f"no zest in {llama_dir}; pass --llama-dir")
if a.kl and not a.text:
    sys.exit(f"--kl needs --text, a plain-text file of at least {a.chunks * 512} tokens")
for m in a.models:
    if not os.path.exists(m):
        sys.exit(f"no such model: {m}")


def run(args, env_extra=None):
    return subprocess.run([zest] + args, capture_output=True, text=True, env=dict(os.environ, **(env_extra or {})))


# XDNA0 is only offered when the NPU passes its check at start. Without it, llama-bench would stop on "invalid
# device" with no reason given.
devices = run(["llama-bench", "--list-devices"])
if "XDNA0" not in devices.stdout + devices.stderr:
    why = next((line.strip() for line in (devices.stdout + devices.stderr).splitlines() if "xdna:" in line), "")
    sys.exit("the NPU add-on isn't available here" + (f": {why}" if why else ""))


def one_round(model, env_extra):
    r = run(["llama-bench", "-m", model, "-dev", "Vulkan0,XDNA0/Vulkan0", "-p", str(a.prompt), "-n", "0",
             "-ub", "2048", "-b", "2048", "-r", "3", "-o", "json"], env_extra)
    found = re.search(r"(?s)\[.*\]", r.stdout)
    if not found:
        return None, r
    tests = json.loads(found.group(0))
    gpu = [t for t in tests if t["n_prompt"] == a.prompt and t["devices"] == "Vulkan0"]
    npu = [t for t in tests if t["n_prompt"] == a.prompt and t["devices"] == "XDNA0/Vulkan0"]
    if not gpu or not npu:
        return None, r
    return (gpu[0]["avg_ts"], npu[0]["avg_ts"], tests[0].get("model_type", "")), r


def kl(model):
    # The GPU's answers first, saved to a file, then the add-on's compared against them. -b 2048 packs four
    # 512-token pieces into each batch, so the add-on gets pieces big enough to take.
    with tempfile.TemporaryDirectory() as tmp:
        common = ["llama-perplexity", "-m", model, "-f", a.text, "-c", "512", "-b", "2048", "-ub", "2048",
                  "--chunks", str(a.chunks), "--kl-divergence-base", os.path.join(tmp, "gpu.kld")]
        g = run(common + ["-dev", "Vulkan0"])
        if g.returncode:
            sys.exit(f"llama-perplexity on the GPU failed (exit {g.returncode})")
        # -ts 0,1: zest adds it only when it chooses the devices itself, and without it llama.cpp's memory fitting
        # divides by zero on mixture-of-experts models
        x = run(common + ["-dev", "XDNA0,Vulkan0", "-ts", "0,1", "--kl-divergence"])
        if x.returncode:
            sys.exit(f"llama-perplexity with the add-on failed (exit {x.returncode})")
    log = x.stdout + x.stderr
    kld = re.search(r"Mean\s+KLD:\s+(-?[\d.]+)", log)
    top = re.search(r"Same top p:\s+([\d.]+)", log)
    return (float(kld.group(1)) if kld else None, float(top.group(1)) if top else None)


results = []
configs = [("add-on", {})] + ([("split off", {"GGML_XDNA_SPLIT": "0"})] if a.no_split else [])
for model in a.models:
    name = os.path.basename(model)
    print(f"=== {name}", flush=True)
    row = {"file": name, "prompt": a.prompt, "rounds": []}
    for r in range(1, a.rounds + 1):
        entry = {}
        for label, env in configs:
            got, proc = one_round(model, env)
            if not got:
                print(f"   round {r}, {label}: no result (exit {proc.returncode})", flush=True)
                continue
            gpu, npu, model_type = got
            # a prompt the NPU failed on was finished by the GPU: not a measurement of the NPU
            failures = len(re.findall("the NPU failed", proc.stderr or ""))
            entry[label] = {"gpu": gpu, "with": npu, "ratio": npu / gpu, "npu_failures": failures}
            row["model"] = model_type or name
            print(f"   round {r}, {label:9s}: GPU alone {gpu:8.1f}  with the add-on {npu:8.1f} tokens/s  {npu / gpu:.2f}x"
                  + (f"  THE NPU FAILED {failures} TIMES" if failures else ""), flush=True)
        row["rounds"].append(entry)
    for label, _ in configs:
        rs = [e[label] for e in row["rounds"] if label in e]
        if rs:
            row[label] = {"gpu_median": statistics.median(x["gpu"] for x in rs),
                          "with_median": statistics.median(x["with"] for x in rs),
                          "ratio_median": statistics.median(x["ratio"] for x in rs),
                          "ratio_lo": min(x["ratio"] for x in rs), "ratio_hi": max(x["ratio"] for x in rs),
                          "n": len(rs)}
    if a.kl:
        row["kld"], row["same_top_pct"] = kl(model)
        print(f"   mean KL divergence {row['kld']}, same top token {row['same_top_pct']}%", flush=True)
        # bit-identical answers mean the add-on took nothing: the GPU measured against itself, not the NPU
        if row["kld"] == 0:
            print(f"   warning: KL is exactly 0, so the add-on took no work; use a longer --text", flush=True)
    results.append(row)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


def cell(row, label):
    # tokens a second, then the slowest and fastest round's ratio to the GPU alone
    s = row.get(label)
    return f"{round(s['with_median']):,} ({s['ratio_lo']:.2f}-{s['ratio_hi']:.2f}x)" if s else "no result"


print()
head = "| model | GPU alone, tokens/s | with the add-on |" + (" split off |" if a.no_split else "") + " rounds |"
rule = "|---|---|---|" + ("---|" if a.no_split else "") + "---|"
if a.kl:
    head += " mean KL divergence | same top token |"
    rule += "---|---|"
print(head)
print(rule)
for row in results:
    s = row.get("add-on")
    line = f"| {row.get('model', row['file'])} | {round(s['gpu_median']):,} | {cell(row, 'add-on')} |" if s else f"| {row['file']} | | no result |"
    if a.no_split:
        line += f" {cell(row, 'split off')} |"
    line += f" {s['n'] if s else 0} |"
    if a.kl:
        line += f" {row.get('kld')} | {row.get('same_top_pct')}% |"
    print(line)
