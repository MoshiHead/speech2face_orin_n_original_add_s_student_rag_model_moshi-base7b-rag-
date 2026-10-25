"""try-rag -- the base7b RAG fine-tune + the Design A causal student driving IMTalker live.

Imports the production live server module read-only, swaps TWO things, and
calls its main().  No production file is copied or edited.

  1. ``MoshiOnlyEngineWithHidden`` -> ``RAGStudentEngine`` (rag_student_engine.py):
     the PersonaPlex **base7b RAG fine-tune** (full 32-layer 7B backbone from
     sadmN/moshi-base7b-rag) served with the single-path streaming RAG protocol
     ported from server_rag_stream.py -- the LM steps every frame on the user's
     audio and generates its own audio + text freely (full duplex, no state
     machine, no forced silence, no barge-in cache reset), the STT runs in
     parallel for the transcript + the mid-pause retrieval, and the 16-token
     fact card is injected into the model's own pause on ``<lookup>``.  The
     engine emits the exact event schema the avatar pipeline consumes (reply
     PCM + ``helium_hidden`` per 80 ms frame).

  2. ``LiveHeliumFMEngine._sample_motion_from_helium`` -> the Design A causal
     motion student (identical to imtalker_try19_student_live.py, imported
     from there so the speculative-rollout safety and the CUDA-graphed student
     step are byte-identical).

The full-7B fast-path hooks of imtalker_try19_student_live.py (int4 trunk,
one-pass Robert_5 depformer, GAN Mimi decoder, TensorRT SEANet, Mimi encoder
CUDA graph) are deliberately NOT installed: they hook the full PersonaPlex
load path (moshi_pkg loaders / pd_parallel / mimi_dec_serve), which the base7b
RAG model does not use -- it loads through moshi_distill's own loader with its
stock depformer and the stock Mimi decoder, exactly as the base7b reference
server (serve_base7b.sh) serves it.
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import torch

import os as _os
IMTALKER = Path(_os.environ.get("IMTALKER_DIR",
    "/home/ubuntu/project/speech2avatar_imtalker_personaplex_try_vad6_8998/IMTalker"))
DESIGN_A = Path(_os.environ.get("DESIGN_A_DIR",
    "/home/ubuntu/project/imtalker_world_model/design_a"))
for _p in (str(IMTALKER), str(DESIGN_A)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Avatar-side patching is shared with try19 (motion student + distilled
# renderer).  Importing try19 only runs its module-level flag reads and
# function definitions; the full-7B hooks live in its main(), which we do
# not call.
from imtalker_try19_student_live import (  # noqa: E402
    load_student, make_student_sampler, load_distilled_renderer,
    load_blink_delta, patch_render)

from rag_student_engine import RAGConfig, RAGStudentEngine  # noqa: E402

LIVE_MODULE = "imtalker_native_passthrough"
LIVE_DIR = str(DESIGN_A / "live")   # for the design_a fork


def _env(name: str, default: str) -> str:
    return _os.environ.get(name, default)



def main():
    pre = argparse.ArgumentParser(add_help=False)
    # -- avatar side (same knobs as imtalker_try19_student_live.py) ----------
    pre.add_argument("--student_ckpt",
                     default=str(DESIGN_A / "checkpoints/design_a_L0_idle_v5/best.pt"))
    pre.add_argument("--live_module", default=LIVE_MODULE,
                     help="module to wrap; imtalker_native_passthrough for the fork")
    pre.add_argument("--blink_delta", default="",
                     help="precomputed pixel-space blink delta (configs/blink_delta.pt)")
    pre.add_argument("--distilled_renderer", default="",
                     help="path to the distilled single-identity renderer checkpoint")
    pre.add_argument("--motion_gain", type=float, default=1.0,
                     help="scale motion deviation to restore articulation damped by ctx_noise")
    pre.add_argument("--blend_frames", type=int, default=10,
                     help="barge-in crossfade length in 25fps frames (10 = 400 ms)")
    pre.add_argument("--jump_thresh", type=float, default=0.15,
                     help="max-abs motion jump that counts as a discontinuity")
    pre.add_argument("--mouth_close", type=float, default=0.5,
                     help="pull the crossfade start toward the reference (closed) pose, 0..1")
    # -- RAG side (the base7b RAG model + retrieval) ---------------------------
    pre.add_argument("--rag_student_ckpt", default=_env("RAG_STUDENT_CKPT", ""),
                     help="dir containing student.pt (rag/ckpt/rag_base7b/step_1500)")
    pre.add_argument("--rag_index", default=_env("RAG_INDEX", ""),
                     help="dir with passages.jsonl + emb.npy (rag/rag_index_dense)")
    pre.add_argument("--rag_tokenizer", default=_env("RAG_TOKENIZER", ""),
                     help="32003-id SentencePiece model (rag/tokenizer.model)")
    pre.add_argument("--rag_retriever", default=_env("RETRIEVER", "hybrid"),
                     choices=("local", "web", "hybrid", "none"))
    pre.add_argument("--rag_web_provider", default=_env("WEB_PROVIDER", "tavily"),
                     choices=("tavily", "brave", "ddg"))
    pre.add_argument("--rag_web_api_key", default=_env("RAG_WEB_API_KEY", "") or None)
    pre.add_argument("--rag_web_timeout", type=float,
                     default=float(_env("WEB_TIMEOUT", "4.0")))
    pre.add_argument("--rag_web_max_results", type=int,
                     default=int(_env("WEB_MAX_RESULTS", "3")))
    pre.add_argument("--rag_web_wait_frames", type=int,
                     default=int(_env("WEB_WAIT_FRAMES", "8")))
    pre.add_argument("--rag_web_trigger_below", type=float,
                     default=float(_env("WEB_TRIGGER_BELOW", "1.1")))
    pre.add_argument("--stt_hf_repo", default=_env("STT_HF_REPO",
                     "kyutai/stt-1b-en_fr-candle"))
    pre.add_argument("--rag_vad_threshold", type=float,
                     default=float(_env("VAD_THRESHOLD", "0.5")))
    pre.add_argument("--rag_max_ref_tokens", type=int,
                     default=int(_env("MAX_REF_TOKENS", "256")))
    pre.add_argument("--rag_ref_tokens_per_slot", type=int,
                     default=int(_env("REF_TOKENS_PER_SLOT", "4")))
    mine, remaining = pre.parse_known_args()

    if any(a == "8998" for a in remaining):
        raise SystemExit("[try-rag] refusing to bind port 8998 - that is the production server")

    import importlib
    if mine.live_module != LIVE_MODULE and LIVE_DIR not in sys.path:
        sys.path.insert(0, LIVE_DIR)

    rag_cfg = RAGConfig(
        student_checkpoint=mine.rag_student_ckpt,
        rag_index=mine.rag_index,
        rag_tokenizer=mine.rag_tokenizer,
        retriever=mine.rag_retriever,
        web_provider=mine.rag_web_provider,
        web_api_key=mine.rag_web_api_key,
        web_timeout=mine.rag_web_timeout,
        web_max_results=mine.rag_web_max_results,
        web_wait_frames=mine.rag_web_wait_frames,
        web_trigger_below=mine.rag_web_trigger_below,
        stt_hf_repo=mine.stt_hf_repo,
        vad_threshold=mine.rag_vad_threshold,
        max_ref_tokens=mine.rag_max_ref_tokens,
        ref_tokens_per_slot=mine.rag_ref_tokens_per_slot,
    )

    live = importlib.import_module(mine.live_module)
    # build_app() constructs MoshiOnlyEngineWithHidden(**avatar_kwargs); the
    # partial injects the RAG config alongside them.
    live.MoshiOnlyEngineWithHidden = functools.partial(
        RAGStudentEngine, rag_cfg=rag_cfg)
    print(f"[try-rag] MoshiOnlyEngineWithHidden -> RAGStudentEngine "
          f"(student={mine.rag_student_ckpt} retriever={mine.rag_retriever} "
          f"provider={mine.rag_web_provider} stt={mine.stt_hf_repo})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_student(mine.student_ckpt, device)

    refx = None
    try:
        refx = torch.load(DESIGN_A / "configs" / "ref_x_3robert.pt",
                          map_location=device)["ref_x"].float().to(device)
    except Exception as e:
        print(f"[try-rag] ref_x unavailable ({e}); mouth_close disabled", flush=True)
    live.LiveHeliumFMEngine._sample_motion_from_helium = make_student_sampler(
        model, cfg, blend_frames=mine.blend_frames, jump_thresh=mine.jump_thresh,
        mouth_close=(mine.mouth_close if refx is not None else 0.0), ref_x=refx,
        motion_gain=mine.motion_gain)
    if mine.distilled_renderer:
        _bl = load_blink_delta(mine.blink_delta, device) if mine.blink_delta else None
        patch_render(live, load_distilled_renderer(mine.distilled_renderer, device), _bl)
    print(f"[try-rag] motion_gain={mine.motion_gain}", flush=True)
    print(f"[try-rag] barge-in crossfade: {mine.blend_frames} frames "
          f"({mine.blend_frames*40} ms), jump_thresh={mine.jump_thresh}, "
          f"mouth_close={mine.mouth_close}", flush=True)
    print("[try-rag] patched _sample_motion_from_helium -> Design A causal student",
          flush=True)
    print(f"[try-rag] wrapped live module: {live.__file__}", flush=True)
    if cfg.lookahead:
        print(f"[try-rag] NOTE lookahead={cfg.lookahead} -> video lags audio by "
              f"{cfg.lookahead*80} ms", flush=True)

    sys.argv = [sys.argv[0]] + remaining
    live.main()


if __name__ == "__main__":
    main()

