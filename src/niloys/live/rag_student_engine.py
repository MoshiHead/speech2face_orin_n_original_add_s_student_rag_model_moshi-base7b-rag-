"""rag_student_engine.py -- base7b RAG speech engine for the IMTalker avatar server.

Drop-in replacement for ``MoshiOnlyEngineWithHidden`` (src/imtalker/liveTry.py)
that runs the PersonaPlex **base7b RAG fine-tune** -- the full 32-layer 7B
backbone from ``sadmN/moshi-base7b-rag`` (``ckpt/rag_base7b/step_1500/student.pt``)
-- with the single-path streaming RAG protocol ported from
``moshi_local/moshi_distill/server_rag_stream.py``:

    user audio -> Mimi codes -> LM (every frame; the model generates its own
                  audio + text freely, full duplex, like server.py)
    STT runs in parallel (non-driving): transcript + VAD; at the user's
                  mid-pause the retrieval fires (local dense + async web search)
    model emits <lookup> -> 16-token fact card injected into its own pause
    -> grounded answer, streamed out frame by frame

The engine speaks the exact event protocol the avatar pipeline
(imtalker_native_passthrough.py) consumes: one dict per 80 ms frame with
``reply_i16_b64`` / ``helium_hidden`` / ``reply_codes`` / RMS + timing fields,
so the talking-head path (Helium -> motion student -> renderer) works
unchanged.  ``helium_hidden`` is the student transformer's own output
(4096-d), captured by a one-line LMGen subclass (_HiddenLMGen).

Serving protocol (identical to the base7b streaming reference server):
  * single path - the LM steps every frame with the user's audio as
                  ``input_tokens`` and generates its own audio + text freely.
                  NO listening/generating state machine, NO forced silence
                  while the user speaks, NO barge-in cache reset: the base7b
                  fine-tune is mute under the old state-machine protocol (the
                  constant forced-silence self-context is a train/serve
                  distribution shift), so the KV cache persists for the whole
                  session and barge-in is just the user's new audio arriving.
  * retrieval   - the STT accumulates the transcript; at the user's mid-pause
                  (half the VAD turn-end silence) prepare_reference fires the
                  local dense retrieval + the async web search, and re-arms
                  <lookup> for the new utterance (multi-turn RAG).
  * <lookup>    - model-driven: the card rides the pause the model produces
                  anyway (text-pause mode), so retrieval adds zero frames /
                  zero silence / zero latency.

Only the serving path is ported; training code is not.  The moshi_distill
package (src/moshi_local) is the same model implementation as
personaplex_rag_code -- same checkpoint, same loaders, same LMGen semantics.
"""

from __future__ import annotations

import base64
import contextlib
import html as html_lib
import json
import logging
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece
import torch
import torchaudio

logger = logging.getLogger(__name__)

TARGET_SR = 24_000          # Mimi sample rate (24 kHz)
FRAME_SIZE = 1_920          # samples per Mimi frame (80 ms @ 24 kHz)

REF_START_ID = 32000
REF_END_ID = 32001
LOOKUP_ID = 32002
APS = 8                     # audio codebooks per stream

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Built-in web-search key, carried over from the RAG reference
# (moshi_local/moshi_distill/server_student_rag.py, itself from moshi/server_w.py).
# Override with RAGConfig.web_api_key or TAVILY_API_KEY.
TAVILY_KEY = "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"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Everything the RAG engine needs that the avatar engine kwargs do not carry."""

    student_checkpoint: str = ""      # step dir containing student.pt (or the .pt file)
    rag_index: str = ""               # dir with passages.jsonl + emb.npy (+ meta.json)
    rag_tokenizer: str = ""           # SentencePiece model with 32003 ids (RAG tokens)
    retriever: str = "hybrid"         # local | web | hybrid | none
    web_provider: str = "tavily"      # tavily | brave | ddg
    web_api_key: str | None = None
    web_timeout: float = 4.0
    web_max_results: int = 3
    web_wait_frames: int = 8          # pause frames spent waiting for an in-flight search
    web_trigger_below: float = 1.1    # search the web when the local top score is below this
    stt_hf_repo: str = "kyutai/stt-1b-en_fr-candle"
    vad_threshold: float = 0.5
    max_ref_tokens: int = 256
    ref_tokens_per_slot: int = 4
    idle_pad_limit: int = 25          # (unused: the streaming protocol has no turn end)
    barge_in_frames: int = 4          # (unused: the streaming protocol has no barge-in reset)
    vad_silence_frames: int = 12      # ~1 s of VAD silence ends the user utterance
    warmup_steps: int = 4


# ---------------------------------------------------------------------------
# Retrieval (dense MiniLM + lexical hybrid + cross-encoder rerank)
# Ported from moshi_local/moshi_distill/server_student_rag.py
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Serving-side retrieval over the grounded_real KB.

    The KB is `passages.jsonl` + `emb.npy`.  Lexical candidates come from the
    same hashed n-gram index used during corpus analysis; the union is
    reranked by a MiniLM cross-encoder.
    """

    def __init__(self, index_dir: str, device: str = "cuda",
                 use_rerank: bool = True, dense_k: int = 50,
                 lexical_k: int = 20):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from moshi_local.moshi_distill.eval_retrieval import build as lex_build
        self.device = device
        self.dense_k = dense_k
        self.lexical_k = lexical_k
        with open(os.path.join(index_dir, "passages.jsonl")) as f:
            self.docs = [json.loads(line)["text"] for line in f]
        self.emb = np.load(os.path.join(index_dir, "emb.npy"))
        assert self.emb.shape[0] == len(self.docs), (self.emb.shape, len(self.docs))
        self.model = SentenceTransformer(ST_MODEL, device=device)
        self.ce = CrossEncoder(RERANK_MODEL, device=device) if use_rerank else None
        self._lidf, self._lpost, self._lnd = lex_build([{"ref": t} for t in self.docs])
        logger.info("[rag] retriever: %d passages | dense=%d lexical=%d rerank=%s",
                    len(self.docs), self.dense_k, self.lexical_k,
                    self.ce is not None)

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        if not query.strip() or not self.docs:
            return []
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        sims = self.emb @ q[0].astype(np.float32)
        cand = np.argsort(-sims)[: self.dense_k].tolist()

        if self.lexical_k:
            from moshi_local.moshi_distill.eval_retrieval import score as lex_score
            from moshi_local.moshi_distill.rag_index_v3 import features as lex_features
            ls = lex_score(self._lidf, self._lpost, self._lnd,
                           lex_features(query))
            seen = set(cand)
            cand.extend(int(i) for i in np.argsort(-ls)[: self.lexical_k]
                        if int(i) not in seen)

        if self.ce is not None and cand:
            scores = self.ce.predict([(query, self.docs[i]) for i in cand],
                                     batch_size=64, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))
            cand = [cand[i] for i in order]

        return [{"text": self.docs[i], "index": int(i),
                 "score": float(sims[i])} for i in cand[:top_k]]

    web_max_results: int = 3
    web_wait_frames: int = 8          # pause frames spent waiting for an in-flight search
    web_trigger_below: float = 1.1    # search the web when the local top score is below this
    stt_hf_repo: str = "kyutai/stt-1b-en_fr-candle"
    vad_threshold: float = 0.5
    max_ref_tokens: int = 256
    ref_tokens_per_slot: int = 4
    idle_pad_limit: int = 25          # 2 s of pads ends the turn
    barge_in_frames: int = 4          # user speech frames that interrupt the model
    vad_silence_frames: int = 12      # ~1 s of VAD silence ends the user utterance
    warmup_steps: int = 4



class WebRetriever:
    """Query -> retrieved passages from a web search API.

    Same return shape as :class:`DenseRetriever` (``{"text", "score", "url"}``)
    so the reference path does not care where the passage came from.  Standard
    library only (``urllib``); always called from a background thread: a search
    takes 0.3-2 s, the frame loop has 80 ms.

    Providers: ``tavily`` (TAVILY_API_KEY), ``brave`` (BRAVE_API_KEY), ``ddg``
    (no key, rate limited, smoke-tests only).
    """

    ENV = {"tavily": "TAVILY_API_KEY", "brave": "BRAVE_API_KEY"}

    def __init__(self, provider: str = "tavily", api_key: str | None = None,
                 timeout: float = 4.0, max_results: int = 3,
                 cache_size: int = 32):
        self.provider = provider
        self.timeout = float(timeout)
        self.max_results = max(1, int(max_results))
        self.api_key = (api_key or os.environ.get(self.ENV.get(provider, ""), "")
                        or (TAVILY_KEY if provider == "tavily" else ""))
        if provider in self.ENV and not self.api_key:
            raise SystemExit(
                f"web search provider {provider!r} needs an API key: pass "
                f"web_api_key or set {self.ENV[provider]}")
        self._cache: dict[str, list[dict]] = {}
        self._cache_size = int(cache_size)

    # -- http ---------------------------------------------------------------
    def _open(self, url: str, headers: dict | None = None,
              data: bytes | None = None) -> bytes:
        req = urllib.request.Request(url, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def _tavily(self, query: str) -> list[dict]:
        body = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }).encode()
        raw = self._open("https://api.tavily.com/search",
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Bearer {self.api_key}"},
                         data=body)
        out = []
        for r in json.loads(raw).get("results", []):
            text = (r.get("content") or r.get("title") or "").strip()
            if text:
                out.append({"text": text, "score": float(r.get("score") or 0.0),
                            "url": r.get("url", "")})
        return out

    def _brave(self, query: str) -> list[dict]:
        url = ("https://api.search.brave.com/res/v1/web/search?"
               + urllib.parse.urlencode({"q": query,
                                         "count": self.max_results}))
        raw = self._open(url, headers={"Accept": "application/json",
                                       "X-Subscription-Token": self.api_key})
        out = []
        for r in json.loads(raw).get("web", {}).get("results", []):
            text = (r.get("description") or "").strip()
            if text:
                out.append({"text": text, "score": 0.0, "url": r.get("url", "")})
        return out

    def _ddg(self, query: str) -> list[dict]:
        url = ("https://html.duckduckgo.com/html/?"
               + urllib.parse.urlencode({"q": query}))
        raw = self._open(url, headers={
            "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }).decode("utf-8", "replace")
        snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', raw, re.S)
        out = []
        for s in snips[: self.max_results]:
            s = html_lib.unescape(re.sub(r"<[^>]+>", " ", s))
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                out.append({"text": s, "score": 0.0, "url": ""})
        if not out:
            logger.warning("duckduckgo returned no results for %r (bot "
                           "challenge / rate limit -- use tavily or brave "
                           "for real use)", query[:60])
        return out

    # -- public -------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        if query in self._cache:
            return self._cache[query][:top_k]
        fn = {"tavily": self._tavily, "brave": self._brave,
              "ddg": self._ddg}.get(self.provider)
        if fn is None:
            raise SystemExit(f"unknown web provider {self.provider!r}")
        try:
            hits = fn(query)
        except Exception as exc:                      # network, quota, parsing
            logger.warning("web search failed (%s): %s", self.provider, exc)
            return []
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[query] = hits
        logger.info("[rag] web search (%s): %d hits for %r", self.provider,
                    len(hits), query[:80])
        return hits[:top_k]


# ---------------------------------------------------------------------------
# Small helpers (ported from server_student_rag.py / server_student.py)
# ---------------------------------------------------------------------------

def decode_stt_tokens(text_tokens: list[torch.Tensor], tokenizer,
                      padding_token_id: int) -> str:
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1).cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpoint loading (ported from server_student_rag.load_rag_student)
# ---------------------------------------------------------------------------

def load_rag_student(path: str | Path, device, dtype):
    from moshi_local.moshi_distill.models.lm import LMModel
    from moshi_local.moshi_distill.ref_adapter import (
        RefAdapter, RefAddAdapter, RefBagAdapter)
    ckpt = Path(path)
    if ckpt.is_dir():
        ckpt = ckpt / "student.pt"
    pkg = torch.load(ckpt, map_location="cpu", weights_only=False)
    kw = dict(pkg["base_kwargs"])
    kw["num_layers"] = len(pkg["kept_layers"])
    model = LMModel(device=device, dtype=dtype, **kw)
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    ref_mode = pkg.get("ref_mode", "latent")
    kind = pkg.get("adapter_kind")
    if kind is None:
        kind = "latent" if pkg.get("adapter") is not None else None
    adapter = None
    if pkg.get("adapter") is not None:
        if kind == "bag":
            cls, extra = RefBagAdapter, {"decay": float(
                pkg.get("ref_decay") or 0.85)}
        elif kind == "add":
            cls, extra = RefAddAdapter, {}
        else:
            cls, extra = RefAdapter, {}
        adapter = cls(model.dim, model.text_emb,
                      n_slots=int(pkg["n_slots"]), **extra)
        adapter = adapter.to(device=device, dtype=dtype)
        adapter.load_state_dict(pkg["adapter"])
        adapter.eval()
    logger.info("[rag] loaded RAG student %s | text_card=%d | ref_mode=%s | "
                "adapter=%s | n_slots=%d | step=%s", ckpt, model.text_card,
                ref_mode, kind, pkg["n_slots"], pkg.get("step"))
    return model, adapter, pkg


class _HiddenLMGenMixin:
    """Stash the transformer output of every step for the avatar's Helium path.

    ``process_transformer_output`` receives the (out_norm'ed) transformer
    output on every LMGen step variant (step / step_with_ref /
    step_embeddings*), so one override covers the whole RAG protocol.
    """

    def process_transformer_output(self, transformer_out, *args, **kwargs):
        self._last_transformer_out = transformer_out
        return super().process_transformer_output(transformer_out, *args, **kwargs)


def _make_hidden_lm_gen():
    from moshi_local.moshi_distill.models.lm import LMGen

    class _HiddenLMGen(_HiddenLMGenMixin, LMGen):
        pass

    return _HiddenLMGen



# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class RAGStudentEngine:
    """base7b RAG fine-tune behind the avatar server's MoshiOnlyEngine interface.

    Constructed by imtalker_native_passthrough.build_app() (patched in place of
    MoshiOnlyEngineWithHidden by imtalker_rag_student_live.py).  The kwargs are
    exactly the ones build_app passes; the full-7B-only ones (moshi_weight,
    quantize_4bit, num_codebooks, context, cfg_coef, moshi_root, mimi_hf_repo,
    placeholder_jpeg_b64) are accepted and ignored -- the base7b RAG model loads
    from RAGConfig.student_checkpoint via moshi_distill's own loader.
    """

    def __init__(self, *, moshi_root: str = "", mimi_hf_repo: str = "",
                 device: str = "cuda", cfg_coef: float = 1.0,
                 placeholder_jpeg_b64: str = "", moshi_weight: str = "",
                 mimi_weight: str = "", tokenizer: str = "",
                 quantize_4bit: bool = False, num_codebooks: int = 8,
                 context: int | None = None, voice_prompt: str = "",
                 voice_prompt_dir: str = "", text_prompt: str = "",
                 rag_cfg: RAGConfig | None = None, **_ignored) -> None:
        from moshi_local.moshi_distill.models import loaders

        cfg = rag_cfg or RAGConfig()
        self.cfg = cfg
        self.device = torch.device(device)
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        self.text_prompt = str(text_prompt or "")
        self.voice_prompt = str(voice_prompt or "")
        self.voice_prompt_dir = str(voice_prompt_dir or "")
        self._pending_events: list[dict] = []
        seed_all(42424242)

        # --- text tokenizer (32003 ids: 32000 + <ref_start>/<ref_end>/<lookup>)
        tok_path = cfg.rag_tokenizer or tokenizer
        if not tok_path or not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"RAG tokenizer not found: {tok_path!r} -- expected the 32003-id "
                "tokenizer.model (rag/tokenizer.model in this repo)")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)
        if self.text_tokenizer.vocab_size() != 32003:
            raise RuntimeError(
                f"RAG tokenizer must have 32003 ids, got "
                f"{self.text_tokenizer.vocab_size()} ({tok_path})")

        # --- Mimi (user-audio encoder + reply decoder; personaplex weights)
        if not mimi_weight:
            from huggingface_hub import hf_hub_download
            mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        logger.info("[rag] loading mimi: %s", mimi_weight)
        self.mimi = loaders.get_mimi(mimi_weight, self.device)
        self.mimi.eval()
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        if self.frame_size != FRAME_SIZE:
            raise RuntimeError(f"expected Mimi frame_size={FRAME_SIZE}, got {self.frame_size}")

        # --- base7b RAG fine-tune (full PersonaPlex backbone + RAG tokens)
        if not cfg.student_checkpoint:
            raise ValueError("RAGConfig.student_checkpoint is required "
                             "(dir with student.pt, e.g. rag/ckpt/rag_base7b/step_1500)")
        logger.info("[rag] loading RAG student: %s", cfg.student_checkpoint)
        student, adapter, pkg = load_rag_student(cfg.student_checkpoint,
                                                 self.device, torch.bfloat16)
        self.adapter = adapter
        self.n_slots = int(pkg["n_slots"])
        self.ref_mode = str(pkg.get("ref_mode", "latent"))
        self.ref_pos = str(pkg.get("ref_pos", "pre_body"))
        self.lookup_id = LOOKUP_ID

        HiddenLMGen = _make_hidden_lm_gen()
        self.lm_gen = HiddenLMGen(
            student, device=self.device,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate, frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False, fast_provided_tail=True)

        # --- STT (kyutai/stt-1b-en_fr-candle via the pip moshi package)
        self._load_stt(cfg)

        # --- retrievers
        self.retriever = None
        if cfg.retriever in ("local", "hybrid"):
            if not cfg.rag_index or not os.path.isdir(cfg.rag_index):
                raise FileNotFoundError(
                    f"RAG index not found: {cfg.rag_index!r} (retriever={cfg.retriever})")
            logger.info("[rag] loading retrieval index: %s", cfg.rag_index)
            self.retriever = DenseRetriever(cfg.rag_index, device=str(self.device))
        self.web_retriever = None
        if cfg.retriever in ("web", "hybrid"):
            self.web_retriever = WebRetriever(cfg.web_provider, cfg.web_api_key,
                                              timeout=cfg.web_timeout,
                                              max_results=cfg.web_max_results)
            logger.info("[rag] web search enabled: provider=%s timeout=%.1fs",
                        self.web_retriever.provider, self.web_retriever.timeout)

        # --- streaming state + prompts + turn state
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.stt_mimi.streaming_forever(1)
        self.stt_lm_gen.streaming_forever(1)
        self._init_turn_state()
        self._warmup(int(cfg.warmup_steps))
        self.reset_session()
        logger.info("[rag] engine ready (student=%s ref_mode=%s/%s voice=%s)",
                    os.path.basename(str(cfg.student_checkpoint)),
                    self.ref_mode, self.ref_pos,
                    self.voice_prompt or "<none>")


    # -- STT loading ---------------------------------------------------------
    def _load_stt(self, cfg: RAGConfig) -> None:
        """Load the STT (transcript + VAD) from the pip ``moshi`` package.

        The STT checkpoint needs the installed 0.2.x package's CheckpointInfo;
        make sure no in-repo ``moshi`` fork shadows it for this import.
        """
        here = os.path.dirname(os.path.abspath(__file__))          # src/niloys/live
        src_root = os.path.dirname(os.path.dirname(here))          # src
        blocked = {os.path.normcase(os.path.join(src_root, "moshi_pkg")),
                   os.path.normcase(os.path.join(src_root, "moshi_local"))}
        orig_path = list(sys.path)
        try:
            sys.path[:] = [p for p in sys.path
                           if os.path.normcase(os.path.abspath(p or os.getcwd()))
                           not in blocked]
            from moshi.models import loaders as stt_loaders
            from moshi.models import LMGen as STTLMGen
        finally:
            sys.path[:] = orig_path
        if not hasattr(stt_loaders, "CheckpointInfo"):
            raise RuntimeError(
                f"the imported moshi package ({stt_loaders.__file__}) has no "
                "CheckpointInfo -- the pip moshi>=0.2.x package is required "
                "for the STT; check that src/moshi_pkg is not on PYTHONPATH")
        logger.info("[rag] STT loaders: %s", stt_loaders.__file__)
        stt_info = stt_loaders.CheckpointInfo.from_hf_repo(cfg.stt_hf_repo)
        self.stt_mimi = stt_info.get_mimi(device=self.device)
        stt_lm = stt_info.get_moshi(device=self.device, dtype=torch.bfloat16)
        stt_lm.eval()
        self.stt_lm_gen = STTLMGen(stt_lm, temp=0, temp_text=0.0)
        self.stt_tokenizer = stt_info.get_text_tokenizer()
        self.stt_padding = int(stt_info.raw_config.get("text_padding_token_id", 3))

    # -- turn state ------------------------------------------------------------
    def _init_turn_state(self) -> None:
        # STT (parallel, non-driving): transcript + VAD utterance tracking
        self._stt_buffer: list[torch.Tensor] = []
        self._stt_in_utterance = False
        self._stt_silence = 0
        # per-utterance RAG state (re-armed at the user's mid-pause)
        self._lookup_done = True           # nothing to look up before the first question
        self._ref_vecs = None
        self._ref_info = None
        self._current_transcript = ""
        self._ref_queue: list[torch.Tensor] = []   # additive ref vecs (ref_mode=add)
        self._text_queue: list[int] = []           # forced card tokens (text/pause)
        self._ref_job: dict | None = None          # async web search for this turn
        self._ref_wait = 0
        self._ref_triggered = False
        self._speech_run = 0

    def _reset_streams(self) -> None:
        for m in (self.mimi, self.lm_gen, self.stt_mimi, self.stt_lm_gen):
            with contextlib.suppress(Exception):
                m.reset_streaming()
        # drop the stashed hidden: the next frame's transformer has not run yet,
        # so a stale hidden from before the reset must not drive the avatar.
        with contextlib.suppress(Exception):
            self.lm_gen._last_transformer_out = None
        self._ref_queue = []
        self._text_queue = []
        self._ref_job = None
        self._ref_wait = 0
        self._ref_triggered = False
        self._speech_run = 0

    # -- session lifecycle (avatar interface) ---------------------------------
    def reset_session(self) -> None:
        """Full reset for a new browser session: streams, turn state, prompts."""
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self._pending_events = []
        self.step = 0
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        self._reset_streams()
        self._init_turn_state()
        self._apply_system_prompts()
        logger.info("[rag] session reset")

    def append_browser_pcm(self, pcm_i16: np.ndarray, input_sr: int) -> None:
        pcm = pcm_i16.astype(np.float32) / 32768.0
        if int(input_sr) != TARGET_SR:
            wav = torch.from_numpy(pcm).view(1, -1)
            pcm = torchaudio.functional.resample(wav, int(input_sr), TARGET_SR)[0].numpy()
        self.input_buffer = np.concatenate([self.input_buffer, pcm.astype(np.float32, copy=False)])

    # -- system prompts ---------------------------------------------------------
    def _resolve_voice_prompt_path(self) -> str:
        if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
            if self.voice_prompt:
                p = os.path.join(self.voice_prompt_dir, self.voice_prompt)
                if os.path.exists(p):
                    return p
                logger.warning("[rag] voice prompt %r missing in %s; falling back",
                               self.voice_prompt, self.voice_prompt_dir)
            cands = sorted(f for f in os.listdir(self.voice_prompt_dir)
                           if f.lower().endswith((".pt", ".wav", ".mp3", ".opus")))
            if cands:
                return os.path.join(self.voice_prompt_dir, cands[0])
        if self.voice_prompt and os.path.isabs(self.voice_prompt) \
                and os.path.exists(self.voice_prompt):
            return self.voice_prompt
        return ""

    @torch.no_grad()
    def _apply_system_prompts(self) -> None:
        voice_path = self._resolve_voice_prompt_path()
        if voice_path:
            if voice_path.endswith(".pt"):
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            else:
                self.lm_gen.load_voice_prompt(voice_path)
            logger.info("[rag] voice prompt: %s", voice_path)
        if self.text_prompt:
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                wrap_with_system_tags(self.text_prompt))
            logger.info("[rag] text prompt loaded: %r", self.text_prompt[:80])
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()

    # -- warmup ------------------------------------------------------------------
    @torch.no_grad()
    def _warmup(self, n_steps: int = 4) -> None:
        t0 = time.perf_counter()
        for _ in range(max(1, int(n_steps))):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32,
                                device=self.device)
            codes = self.mimi.encode(chunk)
            stt_codes = self.stt_mimi.encode(chunk)
            self.stt_lm_gen.step_with_extra_heads(stt_codes)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = self.mimi.decode(tokens[:, 1:9])
        if self.retriever is not None:
            with contextlib.suppress(Exception):
                self.retriever.retrieve("warmup", top_k=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("[rag] warmup done in %.0fms", 1000.0 * (time.perf_counter() - t0))

    # -- token builders (ported from server_student_rag.RAGServerState) ---------
    def _silence(self) -> torch.Tensor:
        from moshi_local.moshi_distill.models.lm import SILENCE_TOKENS
        return torch.as_tensor(SILENCE_TOKENS, dtype=torch.long,
                               device=self.device).view(1, APS, 1)

    def _sine(self, slot: bool = False) -> torch.Tensor:
        from moshi_local.moshi_distill.models.lm import SINE_TOKENS
        t = torch.as_tensor(SINE_TOKENS, dtype=torch.long,
                            device=self.device).view(1, APS, 1).clone()
        if slot:
            # Inner slot frames zero the first *user* codebook (LM cb 9),
            # exactly as `_build_ref_slot_frames` does in training.
            t[:, 0] = -1
        return t

    def _frame_tokens(self, text_id: int, slot: bool = False) -> torch.Tensor:
        """One [1, 17, 1] token frame: text + agent silence + user sine."""
        text = torch.full((1, 1, 1), int(text_id), dtype=torch.long,
                          device=self.device)
        return torch.cat([text, self._silence(), self._sine(slot)], dim=1)

    def _text_tok(self, value: int) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.long, device=self.device)

    # -- reference block injection (ported from server_student_rag) -------------
    @torch.no_grad()
    def inject_ref_block(self, vecs: torch.Tensor) -> int:
        """Force `<ref_start> [K slots] <ref_end>` into the live stream.

        `vecs` is [1, K, dim].  Returns the number of LM frames consumed
        (K + 2) so the caller can emit that much silence.
        """
        lm = self.lm_gen.lm_model
        K = int(vecs.shape[1])
        assert vecs.shape[0] == 1
        if K <= 0:
            return 0
        silence = self._silence()
        sine = self._sine(slot=False)
        slot_user = self._sine(slot=True)
        slot_frame = self._frame_tokens(-1, slot=True)      # emb input for slot j
        emb_dtype = lm.text_emb.weight.dtype
        vecs = vecs.to(dtype=emb_dtype)

        # 1. write <ref_start>; input is the <lookup> frame already in cache
        self.lm_gen.step(input_tokens=sine, moshi_tokens=silence,
                         text_token=self._text_tok(REF_START_ID))
        # 2. write slot 0; input is <ref_start> (no adapter vector)
        self.lm_gen.step(input_tokens=slot_user, moshi_tokens=silence,
                         text_token=self._text_tok(-1))
        # 3. write slot j+1 while feeding slot j + vec[j]   (j = 0 .. K-2)
        for j in range(K - 1):
            emb = lm.embed_codes(slot_frame) + vecs[:, j:j + 1]
            self.lm_gen.step_embeddings_with_tokens(
                emb, input_tokens=slot_user, moshi_tokens=silence,
                text_token=self._text_tok(-1))
        # 4. write <ref_end> while feeding slot K-1 + vec[K-1]
        emb = lm.embed_codes(slot_frame) + vecs[:, K - 1:K]
        self.lm_gen.step_embeddings_with_tokens(
            emb, input_tokens=sine, moshi_tokens=silence,
            text_token=self._text_tok(REF_END_ID))
        return K + 2

    @torch.no_grad()
    def inject_ref_text(self, tokens: list[int]) -> int:
        """Force `<ref_start> tok0 .. tok(K-1) <ref_end>` into the live stream.

        Identical frame budget to :meth:`inject_ref_block` (K + 2 frames).
        Returns the number of frames consumed so the caller emits that much
        silence.
        """
        K = len(tokens)
        if K <= 0:
            return 0
        silence = self._silence()
        sine = self._sine(slot=False)
        slot_user = self._sine(slot=True)
        # write <ref_start>; input is the <lookup> frame already in the cache
        self.lm_gen.step(input_tokens=sine, moshi_tokens=silence,
                         text_token=self._text_tok(REF_START_ID))
        # write token j; input is token j-1 (or <ref_start> for j = 0)
        for t in tokens:
            self.lm_gen.step(input_tokens=slot_user, moshi_tokens=silence,
                             text_token=self._text_tok(int(t)))
        # write <ref_end>; input is the last reference token
        self.lm_gen.step(input_tokens=sine, moshi_tokens=silence,
                         text_token=self._text_tok(REF_END_ID))
        return K + 2

    @torch.no_grad()
    def start_ref_text_pause(self, tokens: list[int]) -> int:
        """Queue the fact card for the next frames; the model keeps speaking.

        Consumes **no frame**: the text stream of the frames the model is
        about to produce is forced to the card ids, exactly as training writes
        them into the pad-text run before the answer.  Their audio is the
        model's own pause, so no silence is added and nothing is delayed.
        """
        self._text_queue = [int(t) for t in tokens if int(t) > 0]
        return len(self._text_queue)

    @torch.no_grad()
    def start_ref_add(self, tokens: list[int]) -> int:
        """Queue additive reference vectors for the next ``len(tokens)`` frames.

        Nothing is written into the cache: the vectors are added to the input
        embeddings of the frames the model is about to speak anyway.
        """
        if not tokens:
            self._ref_queue = []
            return 0
        vecs = self.adapter.encode_ids(list(tokens), self.device)   # [K, dim]
        self._ref_queue = [v.view(1, 1, -1) for v in vecs]
        return int(len(self._ref_queue))


    # -- reference preparation (ported from server_student_rag) -----------------
    def reference_from_hit(self, query: str, hit):
        """Passage hit -> ``(k, block, hit)`` for this checkpoint.

        ``block`` is the kind of thing :meth:`inject_ref_text` /
        :meth:`start_ref_add` / :meth:`start_ref_text_pause` expect: card ids
        in text mode, reference vectors otherwise.
        """
        from moshi_local.moshi_distill.interleaver import compute_ref_slot_count
        from moshi_local.moshi_distill.rag_ref_select import (
            select_ids as select_ref_ids, select_window_ids)
        ref = hit["text"] if hit else ""
        ids = (self.text_tokenizer.encode(ref)[: self.cfg.max_ref_tokens]
               if ref else [])
        if self.ref_mode == "text":
            # The RAG fine-tune was trained on a *contiguous sentence window*
            # of the passage
            # (4-gram Jaccard against the answer), not on the passage head and
            # not on a keyword bag.  Serve the same shape: the window that
            # matches the user's question, `n_slots` tokens (the trained card
            # length).
            k = min(max(int(self.n_slots), 1), len(ids)) if ids else 0
            if k <= 0:
                return 0, [], hit
            query_ids = self.text_tokenizer.encode(query) if query else []
            return k, select_window_ids(self.text_tokenizer, ids, query_ids, k), hit
        k = max(compute_ref_slot_count(len(ids), self.n_slots,
                                       self.cfg.ref_tokens_per_slot), 1)
        if self.ref_mode == "add":
            # No block at all: one vector per fact token over the frames that
            # follow <lookup>, so the "budget" is the card length itself.
            sel = select_ref_ids(self.text_tokenizer, ids,
                                 max(1, int(self.n_slots)))
            return len(sel), sel, hit
        if ids:
            id_t = torch.tensor([ids], dtype=torch.long, device=self.device)
            mask = torch.ones_like(id_t)
            vecs = self.adapter(id_t, mask)[:, :k]
        else:
            vecs = torch.zeros(1, k, self.adapter.dim, device=self.device,
                               dtype=self.adapter.fc1.weight.dtype)
        return k, vecs, hit

    def prepare_reference(self, transcript: str):
        """Retrieve for this turn; web search runs in the background.

        The local index is synchronous (tens of ms); the web search takes
        0.3-2 s and would starve the 80 ms audio loop, so it is started here
        and collected at ``<lookup>`` -- the lead + filler the model speaks in
        between is 2-4 s, which is usually enough.  The local passage is the
        fallback if it is not.
        """
        self._ref_job = None
        self._ref_wait = 0
        if self._is_small_talk(transcript):
            logger.info("[rag] small talk (%r): no retrieval, no card", transcript)
            return 0, [], None
        local = []
        if self.retriever is not None:
            local = self.retriever.retrieve(transcript, top_k=1)
        local_score = float(local[0].get("score", 0.0)) if local else 0.0
        # server_w.py behaviour: the web is a supplement for weak local hits
        # (--web-trigger-below 0.45).  The default (1.1) always searches.
        if (self.web_retriever is not None and (transcript or "").strip()
                and local_score < self.cfg.web_trigger_below):
            job: dict = {"query": transcript, "done": False, "hits": None}

            def _run():
                try:
                    job["hits"] = self.web_retriever.retrieve(transcript, top_k=1)
                except Exception:
                    logger.exception("web retrieval crashed")
                finally:
                    job["done"] = True

            threading.Thread(target=_run, daemon=True,
                             name="web-retrieve").start()
            self._ref_job = job
        return self.reference_from_hit(transcript, local[0] if local else None)

    # Words that carry no retrieval signal: greetings, back-channel, wh-words.
    # A transcript made only of these is chit-chat; injecting a passage there
    # produced things like answering "hello, how are you?" with a movie quote.
    _SMALL_TALK = frozenset("""
        a ah all alright alrighty am an and any are around as at
        back bad be been being bye can cant care cheers clear cool could
        day did do does doing done dont evening ever fine for from
        good goodbye great had has have having hear hearing hello hey hi
        how hows hmm i im in is it its just kind know later let like listen
        listening lot me meet mic microphone mm morning much my name neat
        nice night no nope not now of oh ok okay old on one or our out please
        pretty right see shall should so sound sounds speaking sure talk
        telling test testing thanks thank that thats the then there these they
        thing this those three time to today tomorrow too two four five six
        seven eight nine ten up us very was
        we well were what whats when where which who whom whose why will with
        would yeah yep yes yesterday you your youre
    """.split())

    @classmethod
    def _is_small_talk(cls, text: str) -> bool:
        """True if the transcript has no content word to retrieve on."""
        q = (text or "").strip().lower()
        if not q:
            return True
        if re.search(r"\d", q):
            return False
        words = re.findall(r"[a-z']+", q.replace("'", ""))
        if not words:
            return True
        return not any(w not in cls._SMALL_TALK for w in words)

    @staticmethod
    def _in_pause(pcm) -> bool:
        """True while the model's own frame carries (near) silence."""
        if pcm is None or getattr(pcm, "size", 0) == 0:
            return True
        return float(np.sqrt(np.mean(np.square(pcm)))) < 0.02

    # -- per-frame helpers -------------------------------------------------------
    def _last_hidden(self) -> torch.Tensor | None:
        """The student transformer's output of the latest step, avatar format."""
        h = getattr(self.lm_gen, "_last_transformer_out", None)
        if not isinstance(h, torch.Tensor):
            return None
        return h[:1, -1:].detach().float().cpu()

    def decode_piece(self, token: int) -> str:
        if token in (0, 3) or token >= REF_START_ID or token < 0:
            return ""
        with contextlib.suppress(Exception):
            return self.text_tokenizer.id_to_piece(int(token)).replace("▁", " ")
        return ""

    # -- the 80 ms frame (single-path streaming, ported from server_rag_stream.run_frame) --
    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        """One frame: the LM steps on the user's audio and generates its own
        audio + text freely (full duplex, like server.py); the STT runs in
        parallel for the transcript + the mid-pause retrieval; ``<lookup>``
        (model-driven) injects the fact card into the model's own pause.

        NO listening/generating state machine, NO forced silence while the
        user speaks, NO barge-in cache reset: the base7b RAG fine-tune is mute
        under the old state-machine protocol (its self-context never saw
        constant forced silence in training), so the engine serves it exactly
        like ``server_rag_stream.py`` serves the standalone model.
        """
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))

        t_encode0 = time.perf_counter()
        # --- encode the user's audio for the LM (the only forcing there is) ---
        input_codes = self.mimi.encode(chunk)

        # --- parallel STT (non-driving): transcript + VAD + early retrieval ---
        stt_codes = self.stt_mimi.encode(chunk)
        stt_tokens, vad_heads = self.stt_lm_gen.step_with_extra_heads(stt_codes)
        vad_score = (float(vad_heads[2][0, 0, 0].cpu().item())
                     if vad_heads and len(vad_heads) > 2 else 0.0)
        t_encode1 = time.perf_counter()
        kind, meta = "audio", None
        if stt_tokens is not None:
            self._stt_buffer.append(stt_tokens[:, :1, :].cpu())
            if stt_tokens[0, 0, 0].item() not in (0, 3):
                self._stt_in_utterance = True
            if vad_score > self.cfg.vad_threshold and self._stt_in_utterance \
                    and self._stt_buffer:
                self._stt_silence += 1
                # Mid-pause (half the turn-end silence): the transcript is
                # final enough -- retrieve now so the passage is in hand at
                # <lookup> (server_rag_stream.start_early_web_job timing).
                if self._stt_silence == max(1, self.cfg.vad_silence_frames // 2):
                    transcript = decode_stt_tokens(self._stt_buffer,
                                                   self.stt_tokenizer,
                                                   self.stt_padding)
                    kind, meta = self._prepare_turn_reference(transcript)
            else:
                self._stt_silence = 0
            if self._stt_silence >= self.cfg.vad_silence_frames \
                    and self._stt_in_utterance:
                # Full turn-end pause: close the utterance so the next one
                # starts a fresh transcript (the STT streams keep running).
                self._stt_buffer = []
                self._stt_in_utterance = False
                self._stt_silence = 0

        # --- LM step: user audio in, the model generates its own audio + text ---
        # Drain one queued card token / additive vector per frame, if any.
        ref_vec = self._ref_queue.pop(0) if self._ref_queue else None
        forced_text = self._text_queue.pop(0) if self._text_queue else None
        t_lm0 = time.perf_counter()
        if ref_vec is not None:
            tokens = self.lm_gen.step_with_ref(input_codes, ref_vec=ref_vec)
        elif forced_text is not None:
            # Force the text stream for this frame (the RAG card riding the
            # pause).  Audio is untouched: the model speaks its own frame.
            tokens = self.lm_gen.step(input_codes,
                                      text_token=self._text_tok(forced_text))
        else:
            tokens = self.lm_gen.step(input_codes)
        lm_ms = 1000.0 * (time.perf_counter() - t_lm0)

        if tokens is None:      # warmup frames before the first real step
            return self._event(t0, t_encode0, t_encode1, lm_ms, 0.0,
                               np.zeros(FRAME_SIZE, np.float32), input_rms,
                               0, "", None, self._last_hidden(), "warmup", None)

        text_token = int(tokens[0, 0, 0].item())
        t_decode0 = time.perf_counter()
        # AGENT codebooks only ([:, 1:9]); the user-stream codebooks must not
        # be decoded into the reply voice.
        reply = self.mimi.decode(tokens[:, 1:9])
        reply_pcm = np.clip(reply[0, 0].detach().float().cpu().numpy(), -1.0, 1.0)
        decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
        reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
        helium_hidden = self._last_hidden()

        # Never let card tokens spill into the answer: if the model has
        # started speaking while the card is still draining, drop the rest.
        # Training truncated the card to the pause length too.
        if self._text_queue:
            if self._in_pause(reply_pcm):
                self._speech_run = 0
            else:
                self._speech_run += 1
                if self._speech_run >= 2:
                    logger.info("[rag] card truncated at speech: %d tokens dropped",
                                len(self._text_queue))
                    self._text_queue = []
        else:
            self._speech_run = 0


        # RAG trigger: <lookup> is emitted by the model itself, but the web
        # search may still be in flight.  The lead + filler buys it 2-4 s; if
        # it needs a little more, spend the model's own pause on it.
        if text_token == self.lookup_id and not self._lookup_done:
            self._ref_triggered = True
        if self._ref_triggered and not self._lookup_done:
            job = self._ref_job
            if job is not None and not job["done"]:
                if (self._ref_wait < self.cfg.web_wait_frames
                        and self._in_pause(reply_pcm)):
                    if self._ref_wait == 0:
                        logger.info("[rag] web search still running: spending pause frames")
                    self._ref_wait += 1
                    return self._event(t0, t_encode0, t_encode1, lm_ms, decode_ms,
                                       reply_pcm, input_rms, text_token, "",
                                       reply_codes, helium_hidden, "ref_wait",
                                       {"waited": self._ref_wait})
                self._ref_job = None          # timed out -> local card
            elif job is not None:
                self._ref_job = None
                if job.get("hits") and job["hits"][0].get("text"):
                    _, self._ref_vecs, self._ref_info = self.reference_from_hit(
                        job["query"], job["hits"][0])
                    logger.info("[rag] using web passage %r",
                                job["hits"][0]["text"][:90])
            if self._ref_wait > 0 and not self._in_pause(reply_pcm):
                # we spent pause frames waiting and it is over: answering
                # without a card beats spilling card tokens into the first words
                logger.info("[rag] ref dropped: pause ended before injection")
                self._ref_vecs = None
            n_ref = (int(self._ref_vecs.shape[1])
                     if torch.is_tensor(self._ref_vecs)
                     else len(self._ref_vecs or []))
            self._ref_triggered = False
            self._lookup_done = True
            if n_ref > 0:
                meta = {"frames": 0, "k": n_ref, "hit": self._ref_info}
                if self.ref_mode == "add":
                    # Zero-frame injection: queue the vectors and keep the
                    # stream running.  Only the <lookup> frame itself is muted.
                    k = self.start_ref_add(self._ref_vecs)
                    meta["k"] = k
                    return self._event(t0, t_encode0, t_encode1, lm_ms, decode_ms,
                                       np.zeros(FRAME_SIZE, np.float32), input_rms,
                                       text_token, "", reply_codes, helium_hidden,
                                       "ref_inject", meta)
                if self.ref_mode == "text" and self.ref_pos == "pause":
                    # Card rides the pause the model produces anyway: the
                    # <lookup> frame and every card frame keep their own audio.
                    k = self.start_ref_text_pause(
                        self._ref_vecs if not torch.is_tensor(self._ref_vecs) else [])
                    self._speech_run = 0
                    meta["k"] = k
                    return self._event(t0, t_encode0, t_encode1, lm_ms, decode_ms,
                                       reply_pcm, input_rms, text_token, "",
                                       reply_codes, helium_hidden, "ref_inject", meta)
                # Frame-consuming prefill (latent / text-pre_body): the prefill
                # frames are silent on the wire.
                n_frames = (self.inject_ref_text(self._ref_vecs)
                            if self.ref_mode == "text"
                            else self.inject_ref_block(self._ref_vecs))
                meta["frames"] = n_frames
                return self._event(t0, t_encode0, t_encode1, lm_ms, decode_ms,
                                   np.zeros(FRAME_SIZE, np.float32), input_rms,
                                   text_token, "", reply_codes, helium_hidden,
                                   "ref_inject", meta,
                                   extra_silence_frames=max(0, n_frames - 1))
            logger.info("[rag] no card this turn: answering from the model alone")
        if text_token == self.lookup_id:
            # duplicate / no-reference lookup: never leak the token
            reply_pcm = np.zeros(FRAME_SIZE, np.float32)

        # Forced card tokens are masked from the transcript (server_rag_stream
        # reports 0 for them): the card is grounding, not spoken text.
        token_piece = self.decode_piece(text_token) if forced_text is None else ""
        if token_piece:
            self.audio_text += token_piece
            self.sampled_text += token_piece

        return self._event(t0, t_encode0, t_encode1, lm_ms, decode_ms,
                           reply_pcm, input_rms,
                           text_token if forced_text is None else 0, token_piece,
                           reply_codes, helium_hidden, kind, meta)

    def _prepare_turn_reference(self, transcript: str):
        """Retrieve at the user's mid-pause and re-arm the <lookup> machinery.

        Called when the user is half-way through the turn-end pause (the same
        timing as ``server_rag_stream.start_early_web_job``): the local index
        is synchronous (tens of ms), the web search starts in the background
        and is collected at ``<lookup>``.  Re-arms ``_lookup_done`` so each
        user utterance gets its own card (multi-turn RAG).
        """
        if not (transcript or "").strip():
            return "audio", None
        k, vecs, info = self.prepare_reference(transcript)
        self._ref_vecs = vecs
        self._ref_info = info
        self._current_transcript = transcript
        self._lookup_done = False
        self._ref_triggered = False
        self._ref_wait = 0
        self._ref_queue = []
        self._text_queue = []
        self._speech_run = 0
        logger.info("[rag] transcript=%r k=%s", transcript, k)
        if info:
            logger.info("[rag]   ref score=%.3f %r", info["score"], info["text"][:90])
        return "turn_start", {"transcript": transcript, "k": k, "hit": info}


    # -- event construction (avatar schema) ---------------------------------------
    def _event(self, t0: float, t_encode0: float, t_encode1: float,
               lm_ms: float, decode_ms: float, reply_pcm: np.ndarray,
               input_rms: float, token: int, piece: str,
               reply_codes: torch.Tensor | None,
               helium_hidden: torch.Tensor | None,
               kind: str, meta: dict | None,
               extra_silence_frames: int = 0) -> dict:
        """Build one event dict in the exact schema the avatar pipeline consumes.

        ``extra_silence_frames`` queues that many follow-up silence events (the
        frame-consuming ref-prefill modes); they are drained by
        :meth:`process_ready_steps_limited` before new input is processed.
        """
        reply_pcm = np.asarray(reply_pcm, dtype=np.float32).reshape(-1)
        if reply_pcm.shape[0] < FRAME_SIZE:
            reply_pcm = np.pad(reply_pcm, (0, FRAME_SIZE - reply_pcm.shape[0]))
        elif reply_pcm.shape[0] > FRAME_SIZE:
            reply_pcm = reply_pcm[:FRAME_SIZE]
        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        total_ms = 1000.0 * (time.perf_counter() - t0)
        reply_i16 = (np.clip(reply_pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
        ev = {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": base64.b64encode(reply_i16.tobytes()).decode("ascii"),
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "input_rms": input_rms,
            "token": int(token),
            "piece": piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
            "rag_kind": kind,
        }
        if meta:
            ev["rag_meta"] = {
                k: v for k, v in meta.items() if k != "hit" and v is not None}
            hit = meta.get("hit")
            if hit:
                ev["rag_hit"] = {"score": float(hit.get("score", 0.0)),
                                 "text": str(hit.get("text", ""))[:200]}
        if self.step % 40 == 0 or kind not in ("audio", "silence"):
            logger.info(
                "[rag] step=%d kind=%s token=%d piece=%r in_rms=%.5f "
                "reply_rms=%.5f lm=%.1fms total=%.1fms",
                self.step, kind, int(token), piece, input_rms, reply_rms,
                lm_ms, total_ms)
        for _ in range(max(0, int(extra_silence_frames))):
            self.step += 1
            self._pending_events.append(self._event(
                time.perf_counter(), time.perf_counter(), time.perf_counter(),
                0.0, 0.0, np.zeros(FRAME_SIZE, np.float32), 0.0,
                0, "", None, None, "ref_silence", None))
        return ev

    # -- the avatar-facing drain (avatar interface) -------------------------------
    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process a bounded number of Mimi frames (one per call in production).

        Queued silence events (frame-consuming ref prefill) are drained first,
        then real input frames, mirroring MoshiOnlyEngineWithHidden.
        """
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self._pending_events:
                events.append(self._pending_events.pop(0))
                continue
            if self.input_buffer.shape[0] < FRAME_SIZE:
                break
            pcm = self.input_buffer[:FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events

