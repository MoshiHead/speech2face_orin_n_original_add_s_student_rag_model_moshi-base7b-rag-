#!/usr/bin/env python3
"""tools/test_rag_stream.py -- base7b RAG model smoke test (no avatar, no browser).

Loads the base7b RAG checkpoint with the SAME loader and the SAME single-path
streaming protocol as ``src/niloys/live/rag_student_engine.py`` (the protocol of
``server_rag_stream.py``: the LM steps every frame on the input audio and
generates its own audio + text freely -- no state machine, no forced silence),
feeds it a few seconds of audio, and prints the model's own text.

  * If the model greets / answers (non-pad text), the checkpoint + protocol are
    fine -- any ``[PAD][PAD]...`` flood comes from running the OLD state-machine
    engine (push the updated repo and re-run).
  * If it only ever emits pad here too, the problem is the checkpoint itself
    (or its download), not the serving code.

    $(cat .python_path) tools/test_rag_stream.py                 # 8 s of silence (greeting)
    $(cat .python_path) tools/test_rag_stream.py tools/test_question.wav 12
    $(cat .python_path) tools/test_rag_stream.py <ckpt_dir> <wav> <seconds>
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "niloys", "live"))

import numpy as np
import sentencepiece
import torch

from moshi_local.moshi_distill.models import loaders
from moshi_local.moshi_distill.models.lm import LMGen
from rag_student_engine import load_rag_student, seed_all, wrap_with_system_tags

CKPT = os.path.join(ROOT, "rag", "ckpt", "rag_base7b", "step_1500")
MIMI = os.path.join(ROOT, "weights", "mimi.safetensors")
TOKENIZER = os.path.join(ROOT, "rag", "tokenizer.model")
VOICE = os.path.join(ROOT, "weights", "voices", "Robert_5.pt")
TEXT_PROMPT = os.environ.get(
    "TEXT_PROMPT",
    "You are Robert, an assistant from RB Labs. You have knowledge about crypto, "
    "finance, investment, technology, and general topics. Answer every question "
    "completely, descriptively, and with useful detail.")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    ckpt = args[0] if len(args) > 0 else CKPT
    wav_path = args[1] if len(args) > 1 else ""
    seconds = float(args[2]) if len(args) > 2 else 8.0
    if ckpt.endswith(".wav"):          # first arg was actually the wav
        ckpt, wav_path = CKPT, args[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] ckpt={ckpt}")
    print(f"[test] device={device} torch={torch.__version__}")

    text_tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER)
    assert text_tokenizer.vocab_size() == 32003, \
        f"RAG tokenizer must have 32003 ids, got {text_tokenizer.vocab_size()}"

    mimi = loaders.get_mimi(MIMI, device)
    mimi.eval()
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    t0 = time.perf_counter()
    model, adapter, pkg = load_rag_student(ckpt, device, torch.bfloat16)
    print(f"[test] student loaded in {time.perf_counter()-t0:.1f}s "
          f"(ref_mode={pkg.get('ref_mode')}/{pkg.get('ref_pos')} "
          f"n_slots={pkg['n_slots']} step={pkg.get('step')})")

    lm_gen = LMGen(model, device=device,
                   audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                   sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate,
                   save_voice_prompt_embeddings=False, fast_provided_tail=True)
    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    seed_all(42424242)
    if os.path.exists(VOICE):
        lm_gen.load_voice_prompt_embeddings(VOICE)
        print(f"[test] voice prompt: {VOICE}")
    lm_gen.text_prompt_tokens = text_tokenizer.encode(
        wrap_with_system_tags(TEXT_PROMPT))
    with torch.no_grad():
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
    print("[test] system prompts prefilled -> streaming")

    # input audio: the wav file if given, else silence (the greeting path)
    pcm = np.zeros(0, dtype=np.float32)
    if wav_path:
        import torchaudio
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0)
        if sr != mimi.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, mimi.sample_rate)
        pcm = wav.numpy().astype(np.float32)
        print(f"[test] input: {wav_path} ({pcm.shape[0]/mimi.sample_rate:.1f}s) "
              f"+ {seconds}s of silence")
    pcm = np.concatenate(
        [pcm, np.zeros(int(seconds * mimi.sample_rate), dtype=np.float32)])

    pieces: list[str] = []
    n_pad = n_text = n_lookup = 0
    n_frames = pcm.shape[0] // frame_size
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_frames):
            chunk = torch.from_numpy(
                pcm[i * frame_size:(i + 1) * frame_size]).to(device)[None, None]
            codes = mimi.encode(chunk)
            tokens = lm_gen.step(codes)          # free generation, no forcing
            if tokens is None:
                continue
            t = int(tokens[0, 0, 0].item())
            if t in (0, 3):
                n_pad += 1
            elif t == 32002:
                n_lookup += 1
                pieces.append("<lookup>")
            elif t < 32000:
                n_text += 1
                pieces.append(text_tokenizer.id_to_piece(t).replace("▁", " "))
    dt = time.perf_counter() - t0
    text = "".join(pieces).strip()
    print(f"[test] {n_frames} frames in {dt:.1f}s | pad={n_pad} text={n_text} "
          f"lookup={n_lookup}")
    print(f"[test] TEXT: {text!r}" if text else "[test] TEXT: (pads only)")
    if n_text == 0:
        print("[test] FAIL: the model emitted only pad tokens -- the checkpoint "
              "or its download is the problem (re-check ckpt/rag_base7b/step_1500).")
        sys.exit(1)
    print("[test] OK: the model speaks under the streaming protocol.")


if __name__ == "__main__":
    main()

