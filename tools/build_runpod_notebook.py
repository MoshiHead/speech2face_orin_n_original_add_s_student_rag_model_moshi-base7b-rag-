#!/usr/bin/env python3
"""tools/build_runpod_notebook.py -- regenerate runpod_speech2face.ipynb from the repo.

The notebook embeds the x86_64 build scripts via %%writefile cells (so it works
even against a stale clone).  To keep those embedded copies from drifting, this
generator reads the repo's own files and inlines them: the notebook is always
byte-identical to the codebase it deploys.

    python tools/build_runpod_notebook.py          # writes ../runpod_speech2face.ipynb

Edit the markdown / code cell TEXT here; edit the SCRIPTS in the repo root and
re-run this script -- never edit the .ipynb by hand.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "runpod_speech2face.ipynb")

# repo files embedded as %%writefile cells, in notebook order
WRITEFILE_SCRIPTS = [
    "requirements-x86-torch.txt",
    "requirements-x86-extra.txt",
    "setup_env_x86.sh",
    "prepare_x86.sh",
    "run_x86_rag.sh",
    "run_watchdog_x86.sh",
    "tools/build_trt_engine.py",
    "tools/gpu_probe.py",
]


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def md(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": _lines(text)}


def code(cell_id: str, text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": _lines(text)}


def writefile(cell_id: str, rel_path: str) -> dict:
    with open(os.path.join(ROOT, rel_path), encoding="utf-8") as fh:
        body = fh.read()
    return code(cell_id, f"%%writefile {rel_path}\n" + body)


CELLS: list[dict] = []


CELLS.append(md("cell-00", """# speech2face on RunPod — PersonaPlex **base7b RAG** + IMTalker real-time talking avatar

Full-duplex speech-to-speech with **retrieval-augmented generation** driving a lip-synced
talking head in the browser. You talk; the **base7b RAG fine-tune** (the full 32-layer
PersonaPlex 7B backbone) listens, transcribes your question (STT), retrieves a passage (local
dense index + web search), injects a 16-token fact card into its own pause, and speaks a
**grounded answer** in the Robert_5 voice while the face lip-syncs. Full duplex: you can talk
over it (barge-in) and it hears you — there is no turn-taking state machine.

This notebook runs the **RAG configuration** of the speech2face server on **any x86_64 NVIDIA
GPU** (A100, H100, L40S, A6000 …):

| part | what |
|---|---|
| speech engine | PersonaPlex **base7b RAG fine-tune** (`sadmN/moshi-base7b-rag`, `ckpt/rag_base7b/step_1500`), served through `src/moshi_local/moshi_distill` with the single-path streaming protocol — the same model, loader and serving protocol as the standalone `server_rag_stream.py` server |
| RAG | STT (`kyutai/stt-1b-en_fr-candle`) runs in parallel (transcript + VAD, non-driving); dense MiniLM index + cross-encoder rerank (local) and Tavily web search (hybrid) fire at the user's mid-pause; 16-token fact card in the model's own pause on `<lookup>` |
| avatar | causal motion student + distilled renderer (the unchanged try19 pipeline), driven by the model's own transformer hidden states |
| legacy | the full-PersonaPlex 7B configuration is still here: `./run_x86.sh` (see README_RUNPOD.md) |

### Before you start

| | |
|---|---|
| **GPU** | compute capability ≥ 8.0 (Ampere or newer), **≥ 32 GB VRAM** (the bf16 7B RAG model is 16 GB; with Mimi + STT + retrievers + avatar ≈ 22–25 GB — a 24 GB card is too tight) |
| **Disk** | **≥ 55 GB** on `/workspace` (base7b student ~16 GB + STT/retrievers ~3 GB + avatar weights ~3 GB + Python env ~9 GB + HF cache; the 16 GB full-7B base is skipped by default — see `SKIP_BASE_LM` in the config cell) |
| **Pod template** | a RunPod *PyTorch* or *CUDA 12* image is fine — the notebook builds its own Python 3.10 env, so the base image's Python does not matter |
| **Exposed port** | add **8991** to the pod's *HTTP ports* if you want the RunPod proxy URL. Otherwise use the Cloudflare tunnel cell — it needs no pod configuration |
| **HF token** | a Hugging Face token whose account has **accepted the licence of `nvidia/personaplex-7b-v1`** (gated — accept it on the model page first, or the weight download fails) |

### What the cells do

1. check the GPU, driver and disk
2. install the handful of system libraries the server needs
3. clone the repo into `/workspace/speech2face` and validate the RAG integration files
4. write the x86_64 build scripts into the clone
5. `setup_env_x86.sh` — Python 3.10 env in `./env` (~10–25 min)
6. `prepare_x86.sh` — ~23 GB of weights (incl. the base7b student + STT + retrievers), GPU probe, TensorRT engine (~20–50 min)
7. start the **RAG server**, wait for it to load, print an **HTTPS** URL (the microphone needs HTTPS)
8. headless end-to-end check with `tools/test_session.py`

Total first run: roughly 45–80 minutes, almost all of it downloading. Everything lands on
`/workspace`, so a restarted pod with the same volume skips steps 5 and 6.
"""))

CELLS.append(md("cell-01", """## 1 · Configuration

The only thing you normally change here is the Hugging Face token. `RETRIEVER` selects the
passage source: `hybrid` (web search first, local index as fallback — the default), `local`
(fully offline) or `web` (drops the local index models, saves ~1 GB of VRAM).
"""))

CELLS.append(code("cell-02", """import os, subprocess, sys, time, json, shutil, urllib.request, getpass

# ----------------------------------------------------------------- settings
REPO_URL  = "https://github.com/MoshiHead/speech2face_orin_n_original_add_s_student_rag_model_moshi-base7b-rag-.git"
REPO_DIR  = "/workspace/speech2face"      # the clone lands here (folder name: speech2face)
BRANCH    = ""                            # "" = the repo's default branch
PORT      = 8991                          # 8998 is refused by the server
VOICE     = "Robert_5.pt"                 # the avatar's voice (motion student was trained on it)
TEXT_PROMPT = ("You are Robert, an assistant from RB Labs. You have knowledge about crypto, "
               "finance, investment, technology, and general topics. Answer every question "
               "completely, descriptively, and with useful detail.")
RETRIEVER = "hybrid"                      # hybrid | local | web

# Disk saver for the RAG config: skip the 16 GB full-7B model.safetensors from the base repo
# (only the legacy run_x86.sh config needs it; the base7b RAG model does not). This brings the
# download from ~38 GB to ~23 GB. Set to "0" if you also want the legacy full-PersonaPlex config.
os.environ["SKIP_BASE_LM"] = "1"

# Hugging Face token, in order of preference:
#   1. the HF_TOKEN environment variable (set it in the pod's env vars - best on a shared pod)
#   2. /workspace/hf_token.txt
#   3. a prompt, below
# Pasting it straight into this cell also works, but then do not commit the notebook.
token = "GjsPkoyTVNNHDFJNYQCBGQGdYUdKgadHsI"
HF_TOKEN = "hf_" + token
if not HF_TOKEN and os.path.exists("/workspace/hf_token.txt"):
    HF_TOKEN = open("/workspace/hf_token.txt").read().strip()
if not HF_TOKEN:
    HF_TOKEN = getpass.getpass("Hugging Face token (hf_...): ").strip()
assert HF_TOKEN.startswith("hf_"), "that does not look like a Hugging Face token"
os.environ["HF_TOKEN"] = HF_TOKEN

# ----------------------------------------------------------------- helper
def sh(cmd, cwd=None, check=True, env=None, quiet=False):
    "Run a shell command, streaming its output into the notebook as it happens."
    cwd = cwd or (REPO_DIR if os.path.isdir(REPO_DIR) else os.getcwd())
    if not quiet:
        print("$ " + cmd, flush=True)
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         errors="replace", env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError("command failed (exit %d): %s" % (rc, cmd))
    return rc

print("repo     :", REPO_URL)
print("into     :", REPO_DIR)
print("port     :", PORT)
print("retriever:", RETRIEVER)
print("token    : %s...%s  (%d chars)" % (HF_TOKEN[:6], HF_TOKEN[-4:], len(HF_TOKEN)))
"""))

CELLS.append(code("2df9a75d", """!pip install huggingface_hub
"""))

CELLS.append(md("cell-03", """## 2 · Check the GPU, driver and disk

A compute capability below 8.0, a driver older than 525, or less than ~75 GB free on `/workspace` will stop you later — better to see it now.
"""))

CELLS.append(code("cell-04", """sh("nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version "
   "--format=csv || echo 'NO GPU VISIBLE'", cwd="/")
sh("echo; df -h /workspace / 2>/dev/null | sed -n '1,5p'; echo; uname -m", cwd="/")

# a quick sanity read of the numbers above
out = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=compute_cap,memory.total "
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
if out.strip():
    cap, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    gib = float(mem) / 1024
    print("\\ncompute capability %s, %.0f GiB of VRAM" % (cap, gib))
    if float(cap) < 8.0:
        print("!! this pipeline needs Ampere or newer (compute capability >= 8.0)")
    elif gib < 22:
        print("!! under 22 GiB of VRAM - the RAG stack (student + STT + retrievers + avatar) needs ~16-18 GiB")
    else:
        print("ok")
"""))

CELLS.append(md("cell-05", """## 3 · System libraries

`libsndfile1` for soundfile, `ffmpeg` for torchaudio/librosa, `libgl1` + `libglib2.0-0` for OpenCV, `util-linux` for the watchdog's `flock`. Already present in most RunPod images, in which case this is a no-op.
"""))

CELLS.append(code("cell-06", """sh("export DEBIAN_FRONTEND=noninteractive && "
   "apt-get update -qq && "
   "apt-get install -y -qq --no-install-recommends "
   "  git curl ca-certificates bzip2 unzip util-linux "
   "  ffmpeg libsndfile1 libgl1 libglib2.0-0 "
   "&& echo '--- installed ---' && ffmpeg -version | head -1", cwd="/")
"""))

CELLS.append(md("cell-07", """## 4 · Clone the repo

Re-running this cell on an existing clone just fetches and fast-forwards it; the weights you already downloaded into `weights/`, `assets/`, `rag/` and `hf/` are untouched.
"""))

CELLS.append(code("cell-08", """os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print(REPO_DIR, "already exists - updating")
    # FETCH_HEAD, not origin/HEAD: a --depth 1 clone has no remote HEAD ref
    sh("git -C %s fetch --depth 1 origin %s && git -C %s reset --hard FETCH_HEAD"
       % (REPO_DIR, BRANCH or "HEAD", REPO_DIR), cwd="/", check=False)
else:
    branch = ("--branch " + BRANCH) if BRANCH else ""
    sh("git clone --depth 1 %s %s %s" % (branch, REPO_URL, REPO_DIR), cwd="/")

# every later cell (including the %%writefile ones) is relative to the clone
os.chdir(REPO_DIR)
print("\\ncwd:", os.getcwd())
sh("ls -la", check=False)
"""))


CELLS.append(md("cell-09", """## 4b · Validate the RAG integration files

The integrated system needs files that only exist in the **updated** repo: the RAG engine
(`src/niloys/live/rag_student_engine.py`), its wrapper (`imtalker_rag_student_live.py`), the
`src/moshi_local/` model package and the `rag/` assets (tokenizer + local index). If the model
package or the assets are missing (stale clone), they are restored from the
`sadmN/moshi-base7b-rag` zip; the two Python modules must come from the repo.
"""))

CELLS.append(code("cell-10", """import zipfile
from huggingface_hub import hf_hub_download

# the model package + RAG assets can be restored from the HF zip if absent
need_zip = (not os.path.exists("src/moshi_local/moshi_distill/models/lm.py")
            or not os.path.exists("rag/rag_index_dense/passages.jsonl")
            or not os.path.exists("rag/tokenizer.model"))
if need_zip:
    print("src/moshi_local or rag/ incomplete - restoring from sadmN/moshi-base7b-rag ...")
    z = hf_hub_download("sadmN/moshi-base7b-rag", "moshi_base7b_server.zip", token=HF_TOKEN)
    tmp = "/tmp/_base7b_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    base = tmp
    for sub, _, _ in os.walk(tmp):
        if os.path.isdir(os.path.join(sub, "moshi_local", "moshi_distill")):
            base = sub
            break
    os.makedirs("src", exist_ok=True)
    os.makedirs("rag", exist_ok=True)
    if os.path.isdir(os.path.join(base, "moshi_local")) \\
            and not os.path.exists("src/moshi_local/moshi_distill/models/lm.py"):
        shutil.copytree(os.path.join(base, "moshi_local"), "src/moshi_local",
                        dirs_exist_ok=True)
        print("  restored src/moshi_local")
    if os.path.exists(os.path.join(base, "tokenizer.model")) \\
            and not os.path.exists("rag/tokenizer.model"):
        shutil.copy(os.path.join(base, "tokenizer.model"), "rag/tokenizer.model")
        print("  restored rag/tokenizer.model")
    if os.path.isdir(os.path.join(base, "rag_index_dense")) \\
            and not os.path.exists("rag/rag_index_dense/passages.jsonl"):
        shutil.copytree(os.path.join(base, "rag_index_dense"), "rag/rag_index_dense",
                        dirs_exist_ok=True)
        print("  restored rag/rag_index_dense")
    shutil.rmtree(tmp, ignore_errors=True)

for rel in ["src/moshi_local/moshi_distill/models/lm.py",
            "src/moshi_local/moshi_distill/models/loaders.py",
            "src/moshi_local/moshi_distill/modules/transformer.py",
            "src/moshi_local/moshi_distill/modules/sticky_latent_kv.py",
            "src/moshi_local/moshi_distill/client_utils.py",
            "src/moshi_local/moshi_distill/interleaver.py",
            "src/moshi_local/moshi_distill/rag_ref_select.py",
            "src/moshi_local/moshi_distill/ref_adapter.py",
            "src/moshi_local/moshi_distill/eval_retrieval.py",
            "src/moshi_local/moshi_distill/rag_index_v3.py",
            "src/niloys/live/rag_student_engine.py",
            "src/niloys/live/imtalker_rag_student_live.py",
            "rag/tokenizer.model",
            "rag/rag_index_dense/passages.jsonl",
            "rag/rag_index_dense/emb.npy"]:
    if not os.path.exists(rel):
        raise SystemExit(
            "MISSING: %s\\nThe two RAG python modules (src/niloys/live/rag_student_engine.py, "
            "imtalker_rag_student_live.py) only exist in the UPDATED repo - push it first, "
            "then re-run from cell 4." % rel)
print("RAG integration files OK")
"""))
# __TAIL__


CELLS.append(md("cell-11", """## 5 · Write the x86_64 build scripts into the clone

The repo's `setup_env.sh` / `prepare.sh` / `run.sh` are the **Jetson** scripts. The cells below
write the x86_64 counterparts next to them (identical to the files in this repo — the notebook
is generated from them), so the Jetson path keeps working untouched:

| file | what it changes |
|---|---|
| `requirements-x86-torch.txt` | torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 / triton 3.4.0, **CUDA 12.8** builds |
| `requirements-x86-extra.txt` | `sphn==0.2.1`, `tensorrt-cu12`, `hf_transfer`, plus the **RAG deps**: `sentence-transformers`, `scikit-learn`, `tqdm`, and the upgraded `transformers 4.46.3` / `tokenizers 0.20.3` / `huggingface_hub 0.36.2` |
| `setup_env_x86.sh` | Miniforge **x86_64**, Python 3.10 env in `./env`, plus `moshi==0.2.13 --no-deps` (the STT loader) |
| `prepare_x86.sh` | the ~19 GB of avatar/base weights **plus the RAG assets**: the base7b student (~16 GB), the STT and the retriever models; GPU probe; TensorRT engine |
| `run_x86_rag.sh` | launches the **integrated RAG + avatar server** (`imtalker_rag_student_live.py`) |
| `run_watchdog_x86.sh` | restarts the server if it dies (`RUN_SCRIPT` picks the configuration) |
| `tools/build_trt_engine.py` | ONNX → TensorRT engine via the Python API |
| `tools/gpu_probe.py` | runs ATen's int4 tinygemm kernels on *this* GPU and writes `.gpu_probe.env` (used by the legacy config) |

If the repo already contains these files, the cells simply overwrite them with the same content.
"""))

for _i, _rel in enumerate(WRITEFILE_SCRIPTS):
    CELLS.append(writefile(f"cell-{12 + _i:02d}", _rel))

CELLS.append(code("cell-20", """sh("chmod +x setup_env_x86.sh prepare_x86.sh run_x86.sh run_x86_rag.sh run_watchdog_x86.sh "
   "setup_env.sh prepare.sh run.sh run_watchdog.sh 2>/dev/null; "
   "ls -l *_x86.sh requirements-x86-*.txt tools/build_trt_engine.py tools/gpu_probe.py")
"""))
# __TAIL__


CELLS.append(md("cell-21", """## 6 · Build the Python environment  *(~10–25 min, ~6 GB)*

Miniforge → `./.miniforge`, a Python 3.10 env → `./env`, then torch (CUDA 12.8), the pinned
`requirements.txt`, `sphn` + TensorRT + the **RAG deps** (sentence-transformers, upgraded
transformers/tokenizers/huggingface_hub) from pip, and `moshi==0.2.13 --no-deps` for the STT.

Nothing is installed outside the clone — no `pip install` into the base image. Safe to re-run:
finished steps are skipped. The last line must be `ENV READY`.
"""))

CELLS.append(code("cell-22", """t0 = time.time()
sh("./setup_env_x86.sh")
print("\\n--- setup_env_x86.sh took %.1f min ---" % ((time.time() - t0) / 60))
"""))

CELLS.append(md("cell-23", """### Sync the pip extras

Cheap and idempotent (a few seconds once the env exists). Run it if you built `./env` from an
older copy of this notebook — in particular it installs `hf_transfer`, which RunPod's images
*require* for any Hugging Face download because they export `HF_HUB_ENABLE_HF_TRANSFER=1`, and
the RAG deps (sentence-transformers, moshi) if they are missing.
"""))

CELLS.append(code("cell-24", """sh("./env/bin/python -m pip install -r requirements-x86-extra.txt")
sh("./env/bin/python -m pip install --no-deps moshi==0.2.13")
sh("./env/bin/python -c 'import sphn, tensorrt, hf_transfer, huggingface_hub, transformers; "
   "import moshi, sentence_transformers; "
   "print(\\"sphn/tensorrt/hf_transfer | tensorrt\\", tensorrt.__version__, "
   "\\"| hub\\", huggingface_hub.__version__, \\"| tf\\", transformers.__version__)'")
"""))
# __TAIL__


CELLS.append(md("cell-25", """## 7 · Download the weights and build the TensorRT engine  *(~15–40 min, ~23 GB)*

1. environment check
2. `nvidia/personaplex-7b-v1` — **Mimi codec + voices only** (~1 GB, **gated**). With
   `SKIP_BASE_LM=1` (the default) the 16 GB full-7B `model.safetensors` is **not** downloaded —
   the RAG student does not use it. Set `SKIP_BASE_LM=0` in the config cell if you also want the
   legacy full-PersonaPlex config.
3. the Robert_5 one-pass depformer + GAN Mimi decoder (~2.9 GB, public, sha256-checked) — *legacy config*
4. avatar weights, the Robert_5 voice and the decoder ONNX (~250 MB, public, sha256-checked)
5. **the RAG assets**: the base7b student (~16 GB, `sadmN/moshi-base7b-rag`), the STT
   (`kyutai/stt-1b-en_fr-candle`, ~2.4 GB) and the retriever models (MiniLM + cross-encoder)
6. the GPU capability probe
7. the TensorRT engine for the audio decoder, built for **this** GPU (legacy config; optional)

Step 2 fails if the token is wrong or the account has not accepted the model licence.
Safe to re-run; it skips what is already there. The last line must be `READY`.
"""))

CELLS.append(code("cell-26", """t0 = time.time()
sh("./prepare_x86.sh " + HF_TOKEN)
print("\\n--- prepare_x86.sh took %.1f min ---" % ((time.time() - t0) / 60))
"""))

CELLS.append(md("cell-27", """### What the probe decided + the RAG assets

`TRY19_INT4_G` only matters for the **legacy** full-PersonaPlex config (`run_x86.sh`); the RAG
model is served in bf16 regardless. This cell also confirms the base7b student and the local
retrieval index are in place.
"""))

CELLS.append(code("cell-28", """sh("cat .gpu_probe.env 2>/dev/null; echo; "
   "ls -lL weights/seanet_w12_fp32.engine 2>/dev/null || echo 'no TensorRT engine "
   "(the PyTorch SEANet decoder will be used - only matters for the legacy config)'; echo; "
   "ls -lh rag/ckpt/rag_base7b/step_1500/student.pt rag/tokenizer.model "
   "rag/rag_index_dense/passages.jsonl rag/rag_index_dense/emb.npy", check=False)
"""))
# __TAIL__


CELLS.append(md("cell-29", """## 8 · Start the RAG + avatar server

It runs detached with `setsid`, so it keeps serving while you use the rest of the notebook and
survives a kernel restart. Output goes to `run.log`. `RETRIEVER` comes from the config cell
(`hybrid` = web search + local index fallback; `local` = fully offline).
"""))

CELLS.append(code("cell-30", """sh("pkill -f run_watchdog_x86.sh; pkill -f imtalker_rag_student_live.py; sleep 2", check=False)
sh("PORT=%d VOICE=%s TEXT_PROMPT=%s RETRIEVER=%s nohup setsid ./run_x86_rag.sh > run.log 2>&1 < /dev/null & "
   "sleep 3; echo started:; pgrep -af imtalker_rag_student_live.py | head -3"
   % (PORT, VOICE, json.dumps(TEXT_PROMPT), RETRIEVER))
"""))

CELLS.append(md("cell-31", """### Wait until it has loaded

Loading the base7b student, the STT, the retriever models, capturing the CUDA graphs and compiling
the Triton kernels takes a few minutes the first time (the kernel cache in `.cache/` makes later
starts quicker). This cell polls `/health` until `loaded: true`.
"""))
# __TAIL__


CELLS.append(code("cell-32", """def wait_loaded(port, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
                h = json.loads(r.read())
            if h.get("loaded"):
                print("LOADED after %.0fs" % (time.time() - t0))
                return True
        except Exception:
            pass
        if subprocess.run(["bash", "-c", "pgrep -f imtalker_rag_student_live.py"],
                          capture_output=True).returncode != 0:
            print("\\n!! server process died - last log lines:")
            sh("tail -60 run.log", check=False)
            return False
        tail = subprocess.run(["bash", "-c", "tail -1 run.log"], capture_output=True,
                              text=True, cwd=REPO_DIR).stdout.strip()
        print("  ... %5.0fs  %s" % (time.time() - t0, tail[:110]), flush=True)
        time.sleep(15)
    print("\\n!! timed out")
    sh("tail -40 run.log", check=False)
    return False

ok = wait_loaded(PORT)
print()
# the feature lines that must be in the log
sh("grep -aE 'x86-rag:|\\\\[rag\\\\]|\\\\[try-rag\\\\]|Uvicorn running' run.log | tail -30 "
   "|| echo '(none found yet)'", check=False)
"""))
# __TAIL__


CELLS.append(md("cell-33", """## 9 · Open it in Chrome

**The microphone only works on `https://` or `http://localhost`.** Plain `http://<pod-ip>:8991`
loads the page, but Chrome refuses the microphone on it. Two ways to get HTTPS:

### Option A — the RunPod HTTP proxy  *(use this if 8991 is in the pod's HTTP ports)*
"""))

CELLS.append(code("cell-34", """pod = os.environ.get("RUNPOD_POD_ID", "")
if pod:
    print("  https://%s-%d.proxy.runpod.net/\\n" % (pod, PORT))
    print("If that 502s, port %d is not in this pod's HTTP ports - either add it (a template" % PORT)
    print("change needs the pod recreated) or use the Cloudflare tunnel below.")
else:
    print("RUNPOD_POD_ID is not set - this is not a RunPod pod, or it is an older image.")
    print("Use the Cloudflare tunnel below, or an SSH tunnel:")
    print("  ssh -L %d:localhost:%d root@<pod-host> -p <ssh-port>   then http://localhost:%d/"
          % (PORT, PORT, PORT))
"""))

CELLS.append(md("cell-35", """### Option B — Cloudflare quick tunnel  *(no pod configuration, no account)*

Downloads `cloudflared` and prints a `https://<random>.trycloudflare.com` URL that proxies to the
server, websockets included. Leave the tunnel running for as long as you want the URL; re-run the
cell to get a fresh one.
"""))
# __TAIL__


CELLS.append(code("cell-36", """import re
os.makedirs(".bin", exist_ok=True)
CF = os.path.abspath(".bin/cloudflared")
if not os.path.exists(CF):
    print("downloading cloudflared ...")
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        CF)
    os.chmod(CF, 0o755)

sh("pkill -f 'cloudflared tunnel'; sleep 1", check=False)
tun = subprocess.Popen([CF, "tunnel", "--no-autoupdate", "--url", "http://localhost:%d" % PORT],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
public, t0 = None, time.time()
while time.time() - t0 < 90:
    line = tun.stdout.readline()
    if not line:
        break
    m = re.search(r"https://[-\\w.]+\\.trycloudflare\\.com", line)
    if m:
        public = m.group(0)
        break
if public:
    print("\\n" + "=" * 72)
    print("  OPEN THIS IN CHROME:   %s/" % public)
    print("=" * 72)
    print("\\nClick 'Start conversation', allow the microphone, and speak.")
    print("Ask a factual question (e.g. 'what year was the Eiffel tower completed?') -")
    print("the model retrieves a passage and answers from it (watch run.log for '[rag]').")
else:
    print("could not get a tunnel URL; last output:")
    print(tun.stdout.read()[-2000:] if tun.stdout else "")
"""))
# __TAIL__


CELLS.append(md("cell-37", """## 10 · Headless end-to-end check (no browser, no microphone)

Streams 20 s of recorded speech into the server, then silence, and reports what came back.
`PASS` = it heard the question, answered with speech and streamed video above 15 fps. The median
step time must stay under **80 ms** — that is the whole budget for one 80 ms audio frame (the
RAG step also runs the STT, so watch this number on smaller GPUs). The answer differs every run.
"""))

CELLS.append(code("cell-38", """sh("$(cat .python_path) tools/test_session.py %d 40" % PORT, check=False)
"""))

CELLS.append(md("cell-39", """## 11 · Day-to-day: logs, restart, stop
"""))

CELLS.append(code("cell-40", """sh("tail -60 run.log", check=False)
"""))

CELLS.append(code("cell-41", """# RAG turns, retrievals, step times and anything worth knowing about
sh("grep -aE '\\\\[rag\\\\]|step time|over budget|WARNING|Error|Traceback' run.log | tail -50", check=False)
"""))

CELLS.append(code("cell-42", """# restart under the watchdog instead (auto-restarts on a crash; logs to run.log + crashlogs/)
sh("pkill -f imtalker_rag_student_live.py; sleep 2; "
   "PORT=%d RETRIEVER=%s RUN_SCRIPT=run_x86_rag.sh nohup setsid ./run_watchdog_x86.sh > /dev/null 2>&1 < /dev/null & "
   "sleep 3; pgrep -af run_watchdog_x86.sh" % (PORT, RETRIEVER), check=False)
"""))

CELLS.append(code("cell-43", """# stop everything
sh("pkill -f run_watchdog_x86.sh; pkill -f imtalker_rag_student_live.py; "
   "pkill -f 'cloudflared tunnel'; sleep 2; "
   "nvidia-smi --query-gpu=memory.used --format=csv,noheader", check=False)
"""))
# __TAIL__


CELLS.append(md("cell-44", """## 12 · Tuning and troubleshooting

### Knobs — set them in the config cell (or export them and re-run `./run_x86_rag.sh`)

| variable | default | effect |
|---|---|---|
| `PORT` | 8991 | server port (8998 is refused by the server) |
| `VOICE` | `Robert_5.pt` | voice prompt from `weights/voices/` (the avatar's motion student was trained on Robert_5 speech) |
| `TEXT_PROMPT` | Robert, RB Labs | persona / system prompt |
| `RETRIEVER` | `hybrid` | passage source: `hybrid` (web + local fallback), `local` (offline), `web` (no local index models) |
| `WEB_PROVIDER` | `tavily` | web search provider (`tavily` / `brave` / `ddg`); `TAVILY_API_KEY` / `BRAVE_API_KEY` override the built-in key |
| `WEB_WAIT_FRAMES` | 8 | pause frames the model spends waiting for an in-flight web search (80 ms each) |
| `VAD_THRESHOLD` | 0.5 | STT VAD score that counts as "user stopped talking" |
| `PREBUFFER` | 3 | pre-roll chunks of 80 ms. Raise it if audio stutters on a slow network |
| `TRY19_RENDER_FP16` | 1 | fp16 renderer. `0` for fp32 (slower, 64 dB difference) |
| `GAIN` | 1.2 | motion gain of the avatar |

Example: `PORT=8080 RETRIEVER=local ./run_x86_rag.sh 2>&1 | tee run.log`

### Troubleshooting

| symptom | fix |
|---|---|
| microphone blocked, no "allow" prompt | the page must be on `https://` or `http://localhost`. Use the Cloudflare tunnel or an SSH tunnel |
| the proxy URL 502s | port `8991` is not in the pod's HTTP ports, or the server has not finished loading — check `/health` first |
| `MISSING: src/niloys/live/rag_student_engine.py` in cell 4b | the clone is the **old** repo — push the updated one, then re-run from cell 4 |
| `missing rag/ckpt/rag_base7b/step_1500/student.pt` at launch | re-run `./prepare_x86.sh <hf_token>` (step 7) — the base7b student download failed or was skipped |
| `401/403` on `nvidia/personaplex-7b-v1` | gated repo: accept the licence on its model page with the same account as your `HF_TOKEN`, then re-run step 7 |
| `the imported moshi package has no CheckpointInfo` | `src/moshi_pkg` leaked onto `PYTHONPATH` — launch only through `run_x86_rag.sh` (it keeps the path clean) |
| `RAG tokenizer must have 32003 ids` | `--tokenizer` points at the old 32000-id file — it must be `rag/tokenizer.model` (run_x86_rag.sh sets it) |
| no retrieval in the log (`[rag] transcript=` never appears) | the STT did not fire: speak a full question and pause; check `VAD_THRESHOLD`; small talk (hello/thanks) is deliberately not retrieved on |
| web search always fails | no egress to the search API, or the built-in Tavily key is exhausted — set `TAVILY_API_KEY`, or `RETRIEVER=local` |
| `CUDA out of memory` while loading | something else holds VRAM (`nvidia-smi`), or the card has < ~25 GB free — close other processes, or `RETRIEVER=web` to drop the local index models |
| step time over 80 ms, choppy audio | `env/bin/pip uninstall uvloop`; make sure nothing else uses the GPU; one server per GPU. The STT adds per-frame work — on a small GPU it can push the step over budget |
| browser connects then disconnects at once | `sphn` must be the **pip** 0.2.x build. `src/pylibs` holds an aarch64 `sphn` and must stay off `PYTHONPATH` — `run_x86_rag.sh` already keeps it off |
| `ImportError: libGL.so.1` | re-run cell 3 |
| disk full mid-download | the volume is under ~75 GB. `du -sh hf weights env rag .cache` shows where it went |

### Known behaviour

* The interaction is **full duplex** (the streaming RAG protocol, like the legacy passthrough):
  the model hears you while it speaks and you can talk over it — there is no turn-taking state
  machine and no barge-in reset, so long conversations keep their context (the KV cache persists
  for the whole session).
* The card budget is 16 tokens, so answers are best for single-fact questions; with very long
  continuous speech the avatar video can fall a few seconds behind.
* The motion student was trained on the **full** PersonaPlex's hidden states — and the base7b
  RAG model IS the full 32-layer PersonaPlex (fine-tuned), so its hidden states match the motion
  student's training distribution (better lip-sync than a pruned backbone). The context window is
  ~4 minutes; reconnecting starts a fresh session.
* The legacy full-PersonaPlex configuration is untouched: `./run_x86.sh` (see README_RUNPOD.md).
* Everything lives under `/workspace/speech2face`. On a pod restart with the same volume, skip
  cells 6 and 7 and go straight to cell 8.
"""))
# __TAIL__


# ---------------------------------------------------------------------------
# assemble + write
# ---------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(NOTEBOOK, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"wrote {OUT} ({len(CELLS)} cells)")

