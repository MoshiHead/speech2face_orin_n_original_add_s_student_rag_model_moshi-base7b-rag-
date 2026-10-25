#!/usr/bin/env bash
# run_x86_rag.sh -- PersonaPlex base7b RAG fine-tune + IMTalker real-time avatar on an
# x86_64 NVIDIA GPU (RunPod / Lambda / a local workstation).
#
# The RAG configuration of the speech2face server: the speech engine is the base7b
# RAG fine-tune (full 32-layer PersonaPlex backbone from sadmN/moshi-base7b-rag)
# served with the single-path streaming protocol (full duplex, no state machine,
# no forced silence) with STT + dense/web retrieval and the 16-token fact card
# injected into the model's own pause on <lookup>, and the avatar (motion student
# + distilled renderer + talking-head video) is the unchanged try19 pipeline.
# The legacy full-PersonaPlex configuration is ./run_x86.sh; the Jetson one is ./run.sh.
#
#   ./run_x86_rag.sh                  defaults (port 8991)
#   PORT=8080 ./run_x86_rag.sh        another port (not 8998: refused by the server)
#   RETRIEVER=local ./run_x86_rag.sh  local index only (fully offline, no web search)
#   RETRIEVER=web ./run_x86_rag.sh    web search only (drops the local index models)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"; W="$HERE/weights"; RAG="$HERE/rag"
[ -s "$HERE/.snap_path" ] && [ -s "$HERE/.python_path" ] || { echo "run ./prepare_x86.sh <hf_token> first"; exit 1; }
PY="${PY:-$(cat "$HERE/.python_path")}"   # the Python prepare_x86.sh checked
SNAP="$(cat "$HERE/.snap_path")"
PORT="${PORT:-8991}"   # 8998 is refused by the server (reserved for a legacy production server)
export HF_HOME="${HF_HOME:-$HERE/hf}"

# RunPod (and other) images export HF_HUB_ENABLE_HF_TRANSFER=1; huggingface_hub then refuses to
# download anything at all unless the hf_transfer package is importable. setup_env_x86.sh installs
# it (requirements-x86-extra.txt), but if this Python does not have it, turn the accelerator off
# instead of failing the download.
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" != "0" ] && ! "$PY" -c 'import hf_transfer' 2>/dev/null; then
  echo "    note: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed -> disabling it"
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
# compiled-kernel caches next to the package so they survive on the persistent volume
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/.cache/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}"
export DESIGN_A_DIR="$SRC/niloys" IMTALKER_DIR="$SRC/imtalker"
# PYTHONPATH: src (for the moshi_local package) + src/niloys (live modules).
# NOTE: src/moshi_pkg is deliberately NOT on the path -- the STT is loaded
# through the pip moshi>=0.2.x package, and the RAG engine only uses
# moshi_local.moshi_distill.  src/pylibs stays off too (aarch64 sphn).
export PYTHONPATH="$SRC:$SRC/niloys:${PYTHONPATH:-}"
unset TRY19_DEC_CKPT
export NATIVE_PASSTHROUGH=1
export TRY19_STUDENT_GRAPH="${TRY19_STUDENT_GRAPH:-1}" TRY19_RENDER_FP16="${TRY19_RENDER_FP16:-1}"
export TRY19_CODE_GUARD="${TRY19_CODE_GUARD:-1}"
# torch>=2.6 loads with weights_only=True by default; the RAG student checkpoint
# carries plain python metadata, so keep the old behaviour (we only load our own files).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RAG assets (downloaded / verified by prepare_x86.sh; tokenizer + index ship in the repo)
export RAG_STUDENT_CKPT="${RAG_STUDENT_CKPT:-$RAG/ckpt/rag_base7b/step_1500}"
export RAG_INDEX="${RAG_INDEX:-$RAG/rag_index_dense}"
export RAG_TOKENIZER="${RAG_TOKENIZER:-$RAG/tokenizer.model}"
export RETRIEVER="${RETRIEVER:-hybrid}" WEB_PROVIDER="${WEB_PROVIDER:-tavily}"
[ -s "$RAG_STUDENT_CKPT/student.pt" ] || { echo "missing $RAG_STUDENT_CKPT/student.pt -- re-run ./prepare_x86.sh <hf_token>"; exit 1; }
[ -s "$RAG_TOKENIZER" ] || { echo "missing $RAG_TOKENIZER -- the repo is incomplete (rag/ must contain tokenizer.model)"; exit 1; }
[ -s "$RAG_INDEX/passages.jsonl" ] || { echo "missing $RAG_INDEX/passages.jsonl -- the repo is incomplete (rag/ must contain rag_index_dense/)"; exit 1; }

mkdir -p "$SRC/niloys/live/static" "$HERE/captures"
ln -sfn "$SRC/imtalker/static/assets" "$SRC/niloys/live/static/assets"
ln -sfn "$SRC/niloys/static/index_try21_no_idle_dissolve.html" "$SRC/niloys/live/static/index.html"
exec "$PY" -u "$SRC/niloys/live/imtalker_rag_student_live.py" \
  --live_module imtalker_native_passthrough \
  --student_ckpt "$W/motion_student_38M.pt" --distilled_renderer "$W/renderer_narrow.pt" \
  --motion_gain "${GAIN:-1.2}" \
  --blend_frames 0 --jump_thresh 0.15 --mouth_close 0.5 \
  --host 0.0.0.0 --port "$PORT" \
  --html_path "$SRC/niloys/static/index_try21_no_idle_dissolve.html" \
  --generator_path "$W/generator_last.ckpt" --renderer_path "$W/renderer.ckpt" \
  --adapter_path "$W/adapter_last.pt" \
  --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 \
  --adapter_window_mode lookahead --adapter_future_steps 0 \
  --ref_path "$W/ref.jpeg" --wav2vec_model_path "$W/wav2vec2-base-960h" \
  --moshi_root "$W" --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --mimi_weight "$W/mimi.safetensors" \
  --tokenizer "$RAG_TOKENIZER" \
  --text_prompt "${TEXT_PROMPT:-You are Robert, an assistant from RB Labs. You have knowledge about crypto, finance, investment, technology, and general topics. Answer every question completely, descriptively, and with useful detail.}" \
  --voice_prompt "${VOICE:-Robert_5.pt}" --voice_prompt_dir "$W/voices" \
  --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 1 \
  --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 2 --helium_deque_size 25 \
  --prebuffer_chunks "${PREBUFFER:-3}" --render_sub_batch 2 --renderer_precision fp32 \
  --frame_q_backpressure 3 --buffer_ms 80 --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 \
  --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
  --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise \
  --fp32 --tf32 \
  --silence_helium_path "$W/silence_helium_mean.pt" \
  --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
  --capture_dir "$HERE/captures" \
  --blink_motion_path "$W/blink_motion.pt"

echo "x86-rag: base7b RAG (streaming) + retriever=$RETRIEVER/$WEB_PROVIDER + try19 avatar voice=${VOICE:-Robert_5.pt} port=$PORT"
