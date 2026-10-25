# Running on RunPod (or any x86_64 NVIDIA GPU)

The Orin path (`setup_env.sh` -> `prepare.sh` -> `run.sh`) is unchanged. This page covers the
x86_64 path, which runs the **same model configuration** with the Jetson-only assumptions removed.

There are **two x86_64 configurations**:

| | **RAG** (default, this page) | **legacy** |
|---|---|---|
| speech engine | PersonaPlex **base7b RAG fine-tune** (full 32-layer backbone, `sadmN/moshi-base7b-rag`) + STT + dense/web retrieval + fact card on `<lookup>` | full PersonaPlex 7B (int4 trunk, Robert_5 one-pass depformer, GAN Mimi decoder) |
| launch | `./run_x86_rag.sh` | `./run_x86.sh` |
| interaction | full-duplex streaming (STT in parallel) | full-duplex passthrough |
| VRAM | ~22–25 GB | ~14 GB (int4) / ~26 GB (bf16) |
| notebook | `runpod_speech2face.ipynb` | `runpod_speech2face.ipynb` (same cells; launch `run_x86.sh` instead) |

The RAG configuration is the one `runpod_speech2face.ipynb` launches. Everything below applies
to both unless noted.

```bash
./setup_env_x86.sh                       # once, ~10-25 min: Python 3.10 env in ./env (Miniforge)
./prepare_x86.sh hf_YOUR_TOKEN           # once: ~33 GB of weights, GPU probe, TensorRT engine
export PORT=8991
./run_x86_rag.sh 2>&1 | tee run.log      # RAG + avatar; or ./run_x86.sh for the legacy config
# RUN_SCRIPT=run_x86_rag.sh ./run_watchdog_x86.sh &   # auto-restart on a crash
```

`runpod_speech2face.ipynb` does all of the above from a notebook and also sets up an HTTPS URL.

## The RAG configuration (`run_x86_rag.sh`)

The speech engine is the **base7b RAG fine-tune** from `sadmN/moshi-base7b-rag` — the full
32-layer PersonaPlex 7B backbone fine-tuned to answer **from a retrieved passage**. It is served
through `src/moshi_local/moshi_distill` (the same model implementation as the standalone RAG
server) by `src/niloys/live/rag_student_engine.py`, which is patched in place of the
full-PersonaPlex engine by `src/niloys/live/imtalker_rag_student_live.py`. The avatar (motion
student, distilled renderer, talking-head video) is the **unchanged** try19 pipeline, driven by
the model's own transformer hidden states.

The serving protocol is the **single-path streaming** protocol of `server_rag_stream.py` — the
base7b fine-tune is **mute under the old state-machine protocol** (the constant forced-silence
self-context is a train/serve distribution shift). Per 80 ms frame the LM steps on the user's
audio and generates its own audio + text freely (full duplex, no forced silence, no barge-in
cache reset — the KV cache persists for the whole session). The STT runs in parallel
(transcript + VAD, non-driving); at the user's mid-pause the retrieval fires (local dense +
async web search). On `<lookup>` the 16-token fact card is forced into the text stream of the
pause the model already produces — zero extra frames, zero added latency.

| piece | what |
|---|---|
| student | `rag/ckpt/rag_base7b/step_1500/student.pt` (16 GB, downloaded by `prepare_x86.sh` from `sadmN/moshi-base7b-rag`) |
| tokenizer | `rag/tokenizer.model` (32003 ids: 32000 + `<ref_start>`/`<ref_end>`/`<lookup>`) — ships in the repo |
| local index | `rag/rag_index_dense/` (4998 passages + MiniLM embeddings) — ships in the repo |
| STT | `kyutai/stt-1b-en_fr-candle` (transcript + VAD), loaded via the pip `moshi==0.2.13` package |
| retrievers | `sentence-transformers/all-MiniLM-L6-v2` + `cross-encoder/ms-marco-MiniLM-L-6-v2` (local); Tavily/Brave/DDG (web) |

### RAG environment variables (all optional, shown with defaults)

| variable | default | effect |
|---|---|---|
| `RETRIEVER` | `hybrid` | passage source: `hybrid` (web + local fallback), `local` (offline), `web` (no local index models) |
| `WEB_PROVIDER` | `tavily` | web search provider (`tavily`/`brave`/`ddg`) |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` | built-in dev key | override the web search key |
| `WEB_WAIT_FRAMES` | 8 | pause frames spent waiting for an in-flight web search |
| `VAD_THRESHOLD` | 0.5 | STT VAD score that ends the user's utterance |
| `RAG_STUDENT_CKPT` | `rag/ckpt/rag_base7b/step_1500` | student checkpoint dir |
| `RAG_INDEX` | `rag/rag_index_dense` | local index dir |
| `RAG_TOKENIZER` | `rag/tokenizer.model` | 32003-id tokenizer |

### RAG caveats (read before demoing)

* **Full duplex.** The model hears you while it speaks and you can talk over it — there is no
  turn-taking state machine and no barge-in cache reset, so long conversations keep their
  context (the KV cache persists for the whole session).
* **Lip-sync fidelity.** The motion student was trained on the *full* PersonaPlex's hidden
  states — and the base7b RAG model IS the full 32-layer PersonaPlex (fine-tuned), so its
  hidden states match the motion student's training distribution.
* **Voice.** `Robert_5.pt` is the avatar's voice; if it sounds off, try a voice from the
  personaplex `voices/` dir.
* **Step budget.** The STT adds per-frame work; on small GPUs the 80 ms step budget can be tight.
  Watch `step time` in the log.
* **Single-fact answers.** The card budget is 16 tokens; answers are best for single-fact questions.

## What differs from the Jetson scripts

| | Jetson (`*.sh`) | x86_64 (`*_x86.sh`) |
|---|---|---|
| torch | NVIDIA Jetson index, CUDA 12.6 | `download.pytorch.org/whl/cu128`, torch 2.8.0+cu128 (Ampere / Ada / Blackwell) |
| `sphn` | bundled aarch64 wheel in `src/pylibs` | `sphn==0.2.1` from PyPI (`src/pylibs` is kept **off** `PYTHONPATH`) |
| TensorRT | JetPack, `trtexec` | `tensorrt-cu12` from pip; the engine is built with `tools/build_trt_engine.py` (no `trtexec` in the pip wheels) |
| TensorRT engine | required | **optional** -- without it Mimi's PyTorch SEANet decoder is used (a few ms slower per step, same audio) |
| int4 trunk | always on (`TRY19_INT4_G=32`) | `tools/gpu_probe.py` tries ATen's tinygemm int4 kernels on the actual GPU and writes `.gpu_probe.env`; `run_x86.sh` honours it |

Everything else -- the Robert_5 one-pass depformer, the GAN Mimi decoder, the encoder CUDA
graph, the motion student and the distilled renderer -- is byte-identical to the Orin run.

## Requirements

| what | needed |
|---|---|
| GPU | NVIDIA, compute capability >= 8.0 (Ampere or newer): A100, A6000, L40S, H100. **RAG config: ~22-25 GB of VRAM (>= 32 GB card; a 24 GB card is too tight — the bf16 7B RAG model alone is 16 GB).** Legacy config: ~14 GB with the int4 trunk, ~26 GB if the int4 kernels are unusable on that card |
| driver | >= 525 (CUDA 12.x). `nvidia-smi` must work inside the container |
| disk | ~75 GB free on the volume you clone to (weights ~19 GB + base7b student ~16 GB + STT/retrievers ~3 GB + env ~9 GB + HF cache) |
| Hugging Face | a token whose account has **accepted the licence of `nvidia/personaplex-7b-v1`** |
| network | github.com, conda-forge, PyPI, download.pytorch.org, huggingface.co (+ the web-search API if `RETRIEVER=hybrid/web`) |
| system packages | `ffmpeg libsndfile1 libgl1 git curl` (the RunPod PyTorch images already have most of them) |

## Microphone needs HTTPS

Chrome only grants microphone access on `https://` or `http://localhost`. On a pod:

* **RunPod HTTP proxy** -- expose the port in the pod template, then open
  `https://<POD_ID>-8991.proxy.runpod.net/`. Websockets are proxied.
* **Cloudflare quick tunnel** -- no pod configuration and no account:
  `cloudflared tunnel --url http://localhost:8991` prints a `https://<random>.trycloudflare.com` URL.
* **SSH tunnel** -- `ssh -L 8991:localhost:8991 root@<pod> -p <port>`, then `http://localhost:8991/`.

## Verify it works without a browser

```bash
until curl -s http://localhost:8991/health | grep -q '"loaded":true'; do sleep 10; done
$(cat .python_path) tools/test_session.py 8991 40
```

`PASS` means it heard the question, answered with speech and streamed video above 15 fps.
The step time printed must stay under 80 ms for real time.

The log should contain (RAG configuration):

```
x86-rag: base7b RAG (streaming) + retriever=hybrid/tavily + try19 avatar voice=Robert_5.pt port=8991
[try-rag] MoshiOnlyEngineWithHidden -> RAGStudentEngine (student=... retriever=hybrid ...)
[rag] loaded RAG student .../rag/ckpt/rag_base7b/step_1500/student.pt | text_card=32003 | ref_mode=text | ...
[rag] STT loaders: .../site-packages/moshi/models/loaders.py
[rag] retriever: 4998 passages | dense=50 lexical=20 rerank=True
[rag] web search enabled: provider=tavily timeout=4.0s
[rag] warmup done in ...ms
[rag] engine ready (student=step_1500 ref_mode=text/pause voice=Robert_5.pt)
[try19] student CUDA-graphed ...
INFO:     Uvicorn running on http://0.0.0.0:8991
```

For the **legacy** configuration (`run_x86.sh`) the log instead shows the `x86: trunk int4 ...`,
`[try19] int4 G=32`, `TRY19_PD hook`, `Mimi decoder fine-tune hook` and `trt_seanet` lines
(the `int4`/`trt_seanet` lines are absent when the GPU probe disabled int4 or no engine was built).

## Troubleshooting

| symptom | fix |
|---|---|
| `this script is for x86_64` | you are on a Jetson: use `setup_env.sh` / `prepare.sh` / `run.sh` |
| `nvidia-smi not found` | the pod was started without a GPU, or without the NVIDIA runtime |
| `int4 tinygemm : UNAVAILABLE` in the probe | normal on some architectures. The trunk then runs bf16 and needs ~26 GB of VRAM; pick a bigger GPU or accept the slower path |
| `CUDA out of memory` at load | another process is holding VRAM (`nvidia-smi`), or the card is too small for the plan the probe chose |
| `hf_transfer package is not available` | the image exports `HF_HUB_ENABLE_HF_TRANSFER=1`. `pip install -r requirements-x86-extra.txt` into `./env` (or re-run `./setup_env_x86.sh`); `prepare_x86.sh` also disables the accelerator itself when the package is missing |
| step 2 of `prepare_x86.sh` fails | token wrong, or the HF account has not accepted the `nvidia/personaplex-7b-v1` licence |
| no TensorRT engine | see `weights/trtexec_build.log`. Not fatal; the PyTorch SEANet decoder is used |
| browser connects then disconnects | `sphn` must be 0.2.x and must be the **pip** one -- `src/pylibs` must not be on `PYTHONPATH` on x86 |
| microphone blocked | the page must be on HTTPS or `localhost`; see above |
| choppy audio, step time > 80 ms | `pip uninstall uvloop`; make sure nothing else uses the GPU; one server per GPU. The RAG STT adds per-frame work — on a small GPU it can push the step over budget |
| `ImportError: libGL.so.1` | `apt-get install -y libgl1` |
| `missing rag/ckpt/rag_base7b/step_1500/student.pt` at launch | re-run `./prepare_x86.sh <hf_token>` — the base7b student download failed or was skipped |
| the model only emits `[PAD]` / stays mute | you are serving the base7b checkpoint with the **old state-machine engine** — the base7b fine-tune needs the streaming protocol (it never saw forced-silence self-context in training). Pull the updated repo: `rag_student_engine.py` now uses the `server_rag_stream.py` protocol |
| `the imported moshi package has no CheckpointInfo` | `src/moshi_pkg` leaked onto `PYTHONPATH` — launch only through `run_x86_rag.sh` (it keeps the path clean) |
| `RAG tokenizer must have 32003 ids` | `--tokenizer` points at the old 32000-id file — it must be `rag/tokenizer.model` (`run_x86_rag.sh` sets it) |
| no `[rag] transcript=` in the log | the STT did not fire: speak a full question and pause; small talk (hello/thanks) is deliberately not retrieved on |
| web search always fails | no egress to the search API, or the built-in Tavily key is exhausted — set `TAVILY_API_KEY`, or `RETRIEVER=local` |
