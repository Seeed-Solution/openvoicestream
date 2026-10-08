"""Measure end-to-end latency on the unified WS `/v2v/stream` endpoint.

Complements `bench/measure_v2v.py`, which measures the legacy split
pipeline (POST /asr/stream → POST /tts/stream). This script exercises
the new unified WebSocket and reports:

  • tfd_ms                — first PCM chunk sent → first asr_partial
  • stop_to_endpoint_ms   — last PCM chunk → asr_endpoint  (VAD pre-fire)
  • stop_to_final_ms      — last PCM chunk → asr_final
  • final_to_tts_audio_ms — asr_final → first TTS PCM frame
  • stop_to_tts_audio_ms  — last PCM chunk → first TTS PCM frame (E2E)

Multi-utterance mode (`--multi`) measures per-utterance stop→final by
playing the same WAV N times back-to-back with synthesized silence
between them. Each utterance must produce its own mid-session
asr_final with session_complete=false; the final asr_eos triggers
session_complete=true.

Usage:
    uv run --with websocket-client --with soundfile --with numpy \
        python bench/measure_v2v_unified.py \
        --host 100.89.94.11:8621 \
        --wav tests/data/short_zh.wav \
        --runs 5 \
        --tts            # enable round-trip TTS measurement
        --multi 3        # play wav 3x in one session

Output: JSON to stdout (one record per run) + summary table to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import statistics
import struct
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import soundfile as sf
import websocket


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav_16k_i16(path: str) -> tuple[bytes, float]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        )
    duration = len(audio) / 16000
    i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    return i16, duration


def make_silence(ms: int) -> bytes:
    n = int(16000 * ms / 1000)
    return b"\x00" * (n * 2)


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """Latencies for one utterance within a session."""
    tfd_ms: float | None = None
    stop_to_endpoint_ms: float | None = None
    stop_to_final_ms: float | None = None
    final_text: str = ""
    session_complete: bool | None = None


@dataclass
class RunResult:
    audio_dur_s: float
    utterances: list[Utterance] = field(default_factory=list)
    final_to_tts_audio_ms: float | None = None
    stop_to_tts_audio_ms: float | None = None
    # Server-loop mode only: the reply is produced by the server's own LLM, so
    # asr_final → tts_started isolates the LLM leg (TTFT + tokens up to the
    # first TTS-ready clause) and tts_started → first PCM isolates synthesis.
    final_to_tts_started_ms: float | None = None
    tts_started_to_audio_ms: float | None = None
    assistant_text: str = ""
    client_llm_ttft_ms: float | None = None
    tts_pcm_bytes: int = 0
    tts_sample_rate: int | None = None
    tts_last_pcm_bytes: int = 0
    tts_last_pcm_ms: float | None = None
    tts_complete: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(
    host: str,
    wav_pcm: bytes,
    audio_dur_s: float,
    *,
    language: str = "Chinese",
    vad: str = "silero",
    tts_enabled: bool = False,
    tts_language: str = "auto",
    multi_count: int = 1,
    silence_ms: int = 700,
    realtime: bool = True,
    chunk_ms: int = 100,
    timeout: float = 30.0,
    prepare_lead_ms: int = 0,
    server_loop: bool = False,
    client_llm: bool = False,
    ws_factory: Callable[..., Any] | None = None,
    llm_backend_factory: Callable[[], Any] | None = None,
) -> RunResult:
    """One V2V session. If multi_count > 1, plays the wav N times with
    silence_ms gaps between, expecting N mid-session finals + 1 closing
    final."""
    if client_llm and (server_loop or multi_count != 1):
        raise ValueError("--client-llm requires one single-turn audio input and no server loop")
    # Client-owned LLM still needs the persistent multi-utterance wire
    # protocol for a single audio turn: single mode closes the websocket as
    # soon as it emits asr_final, before the client can send text/tts_flush.
    multi_utterance = multi_count > 1 or client_llm
    cfg: dict[str, Any] = {
        "type": "config",
        "asr_language": language,
        "sample_rate": 16000,
        "vad": vad,
        "vad_silence_ms": 500,
        "multi_utterance": multi_utterance,
    }
    if tts_enabled:
        cfg["tts_language"] = tts_language

    result = RunResult(audio_dur_s=audio_dur_s * multi_count)

    ws_url = f"ws://{host}/v2v/stream"
    ws = (ws_factory or websocket.create_connection)(ws_url, timeout=timeout)
    client_llm_thread: threading.Thread | None = None
    client_llm_ctl: dict[str, Any] = {
        "loop": None,
        "task": None,
        "backend": None,
        "cancel_requested": False,
    }
    try:
        ws.send(json.dumps(cfg))

        chunk_bytes = int(16000 * chunk_ms / 1000) * 2
        silence_pcm = make_silence(silence_ms)
        chunk_dur_s = chunk_ms / 1000.0

        # Per-utterance state for sender side
        utt_pending = [Utterance() for _ in range(multi_count)]
        utt_idx_send = 0   # which utterance is currently being sent
        t_stops: list[float] = []        # filled as each utterance's audio ends
        t_first_send: float | None = None
        t_session_stop: float | None = None  # last byte of last utterance

        # Sender loop: send all utterances, recording stop time per utterance.
        # The reader runs in the main thread interleaved with sends via short
        # socket timeouts (same pattern as bench/perf/client.py).
        def send_audio(pcm: bytes, mark_stop_for: int | None = None):
            nonlocal t_first_send
            for off in range(0, len(pcm), chunk_bytes):
                chunk = pcm[off:off + chunk_bytes]
                if not chunk:
                    break
                ws.send_binary(chunk)
                if t_first_send is None:
                    t_first_send = time.monotonic()
                if realtime:
                    time.sleep(chunk_dur_s)
                # Opportunistic read for partial / endpoint frames while
                # streaming audio (so tfd / endpoint measurements include
                # mid-stream events).
                drain_nonblocking()
            if mark_stop_for is not None:
                t_stops.append(time.monotonic())

        # ── reader state (shared by the turn gate below and the final drain) ──
        got_session_complete = False     # set True on asr_final (single mode)
                                         # or session_complete=true (multi mode)
        utt_idx_recv = 0
        t_final_for_tts: float | None = None
        t_tts_started: float | None = None
        tts_audio_first: float | None = None
        tts_done = not tts_enabled and not client_llm
        t_last_audio: float | None = None   # last TTS binary frame seen
        stop_reading = False
        client_llm_called = False
        client_llm_done = False
        client_turn_final = False
        client_llm_started = 0.0
        client_llm_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def start_client_llm(user_text: str) -> None:
            nonlocal client_llm_called, client_llm_started, client_llm_thread
            client_llm_called = True
            client_llm_started = time.monotonic()

            def worker() -> None:
                loop = asyncio.new_event_loop()
                client_llm_ctl["loop"] = loop

                async def run() -> None:
                    backend = None
                    if llm_backend_factory is not None:
                        backend = llm_backend_factory()
                    else:
                        try:
                            from ovs_agent.llm.openai_compat import OpenAICompatBackend
                        except ModuleNotFoundError as exc:
                            if exc.name != "ovs_agent":
                                raise
                            from agent.ovs_agent.llm.openai_compat import OpenAICompatBackend
                        backend = OpenAICompatBackend(
                            base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
                            model=os.environ.get("LLM_MODEL", "qwen3.5-flash"),
                        )
                    client_llm_ctl["backend"] = backend
                    try:
                        messages = [
                            {"role": "system", "content": os.environ.get(
                                "SYSTEM_PROMPT", "Reply briefly in the user's language."
                            )},
                            {"role": "user", "content": user_text},
                        ]
                        chunks: list[str] = []
                        async for token in backend.stream(messages):
                            if token:
                                chunks.append(token)
                                client_llm_queue.put(("token", (token, time.monotonic())))
                        text = "".join(chunks)
                        if not text:
                            raise RuntimeError("empty assistant text")
                        client_llm_queue.put(("done", text))
                    finally:
                        close = getattr(backend, "aclose", None)
                        if close is None:
                            client = getattr(backend, "client", None)
                            close = getattr(client, "close", None)
                        if close is not None:
                            try:
                                await close()
                            except Exception:
                                pass

                async def bounded_run() -> None:
                    await asyncio.wait_for(run(), timeout=timeout)

                try:
                    task = loop.create_task(bounded_run())
                    client_llm_ctl["task"] = task
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    if not client_llm_ctl["cancel_requested"]:
                        client_llm_queue.put(("error", "CancelledError"))
                except Exception as exc:
                    # Never copy provider exception text into results: it can
                    # contain URLs, request headers, or other credentials.
                    client_llm_queue.put(("error", type(exc).__name__))
                finally:
                    loop.close()

            client_llm_thread = threading.Thread(target=worker, daemon=True)
            client_llm_thread.start()

        def drain_client_llm() -> None:
            nonlocal client_llm_done, stop_reading
            while True:
                try:
                    kind, payload = client_llm_queue.get_nowait()
                except queue.Empty:
                    return
                if kind == "token":
                    token, token_at = payload
                    if result.client_llm_ttft_ms is None:
                        result.client_llm_ttft_ms = (token_at - client_llm_started) * 1000
                    ws.send(json.dumps({"type": "text", "text": token}))
                elif kind == "done":
                    result.assistant_text = payload
                    ws.send(json.dumps({"type": "tts_flush"}))
                    client_llm_done = True
                else:
                    result.error = f"client LLM failed: {payload}"
                    client_llm_done = True
                    stop_reading = True
                    return

        def slot() -> Utterance:
            """Current utterance slot, clamped.

            multi_utterance emits N mid-session finals (session_complete=false)
            PLUS a closing one, i.e. N+1 finals for N slots, so the receive index
            legitimately reaches len(utt_pending). Clamping keeps the closing
            final landing on the last slot instead of raising IndexError.
            """
            return utt_pending[min(utt_idx_recv, len(utt_pending) - 1)]

        def handle(msg) -> None:
            """Dispatch one frame. Sets ``stop_reading`` on a fatal frame."""
            nonlocal got_session_complete, utt_idx_recv, t_final_for_tts
            nonlocal t_tts_started, tts_audio_first, tts_done, stop_reading
            nonlocal t_last_audio, client_turn_final
            if isinstance(msg, bytes):
                t_last_audio = time.monotonic()
                if client_llm:
                    if result.tts_sample_rate is None:
                        if len(msg) < 4:
                            result.error = "tts audio frame missing sample-rate header"
                            stop_reading = True
                            return
                        result.tts_sample_rate = struct.unpack("<I", msg[:4])[0]
                        payload = msg[4:]
                    else:
                        payload = msg
                    if payload:
                        result.tts_pcm_bytes += len(payload)
                        result.tts_last_pcm_bytes = len(payload)
                        if t_session_stop is not None:
                            result.tts_last_pcm_ms = (time.monotonic() - t_session_stop) * 1000
                if tts_enabled and tts_audio_first is None and len(msg) > 4:
                    # First TTS binary frame: 4-byte sample rate header.
                    tts_audio_first = time.monotonic()
                    if t_final_for_tts is not None:
                        result.final_to_tts_audio_ms = (tts_audio_first - t_final_for_tts) * 1000
                    # In multi-utterance mode the first reply audio arrives
                    # DURING the sender loop, while t_session_stop is still
                    # unassigned (it is only set after the loop). Fall back to
                    # the most recent utterance stop, which is the turn this
                    # reply belongs to — so this stays a per-turn mouth-to-ear
                    # figure rather than silently reporting nothing.
                    ref_stop = t_session_stop if t_session_stop is not None else (
                        t_stops[-1] if t_stops else None)
                    if ref_stop is not None:
                        result.stop_to_tts_audio_ms = (tts_audio_first - ref_stop) * 1000
                    if t_tts_started is not None:
                        result.tts_started_to_audio_ms = (tts_audio_first - t_tts_started) * 1000
                return
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                return
            t = data.get("type")
            now = time.monotonic()
            if t == "asr_partial":
                u = slot()
                if u.tfd_ms is None and t_first_send is not None:
                    u.tfd_ms = (now - t_first_send) * 1000
            elif t == "asr_endpoint":
                u = slot()
                if utt_idx_recv < len(t_stops):
                    u.stop_to_endpoint_ms = (now - t_stops[utt_idx_recv]) * 1000
            elif t == "asr_final":
                u = slot()
                complete = data.get("session_complete")
                # The closing final lands on the last slot too (N+1 finals for N
                # slots, see slot()). Writing it through would recompute that
                # utterance's stop_to_final_ms from the SAME t_stops entry but at
                # the closing final's arrival — inflating the last utterance's
                # ASR latency by however long the session took to wind down. So
                # a slot's metrics are written once; later finals only advance
                # the state machine.
                already_measured = u.stop_to_final_ms is not None
                if not already_measured:
                    u.final_text = data.get("text", "") or ""
                    u.session_complete = complete
                    if utt_idx_recv < len(t_stops):
                        u.stop_to_final_ms = (now - t_stops[utt_idx_recv]) * 1000
                if t_final_for_tts is None:
                    # First final triggers TTS in our test harness (when
                    # tts_enabled): echo it back as text. In server-loop mode
                    # the server runs the LLM itself and starts TTS on its own
                    # — injecting text here would bypass the LLM entirely.
                    t_final_for_tts = now
                    if client_llm and u.final_text:
                        start_client_llm(u.final_text)
                    elif tts_enabled and u.final_text and not server_loop:
                        ws.send(json.dumps({"type": "text", "text": u.final_text}))
                        ws.send(json.dumps({"type": "tts_flush"}))
                if multi_utterance:
                    if complete:
                        got_session_complete = True
                    elif client_llm:
                        # This is the one turn's final, not a session close.
                        # The websocket remains open until the per-turn
                        # tts_done arrives; closing it here reproduces the
                        # single-utterance server behavior and loses audio.
                        client_turn_final = True
                    elif utt_idx_recv < len(utt_pending) - 1:
                        utt_idx_recv += 1
                    else:
                        # Already on the last slot: the closing final will land
                        # here too rather than running off the end.
                        got_session_complete = True
                else:
                    got_session_complete = True
            elif t == "tts_started":
                if result.final_to_tts_started_ms is None and t_final_for_tts is not None:
                    result.final_to_tts_started_ms = (now - t_final_for_tts) * 1000
                t_tts_started = now
            elif t == "tts_done":
                tts_done = True
                if client_llm:
                    result.tts_complete = True
            elif t == "error":
                result.error = data.get("error", "(no detail)")
                stop_reading = True

        def drain_nonblocking() -> None:
            """Consume whatever is queued without blocking, through ``handle``.

            This MUST route every frame into the same state machine the main
            drain uses. The previous version had its own partial copy of the
            dispatch logic and threw binary frames away with the comment "TTS
            audio handled by main loop" — which is false once this is called
            while audio is already in flight: the frames are consumed HERE, so
            the main loop never sees them. In multi-utterance server-loop mode
            that silently ate all the reply PCM *and* the per-turn tts_started /
            tts_done, which then looked like "the server never sends audio on
            later turns" and cost two wrong diagnoses.
            """
            drain_client_llm()
            ws.settimeout(0.001)
            while True:
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    return
                if msg is None or msg == "":
                    return
                handle(msg)
                drain_client_llm()

        def pump_until(done, budget_s: float, mark_error: bool = True) -> None:
            """Read + dispatch frames until ``done()`` or the budget expires.

            ``mark_error=False`` lets a caller treat budget exhaustion as a
            pacing hint rather than a failed measurement (no ``result.error``).
            """
            nonlocal stop_reading
            deadline = time.monotonic() + budget_s
            ws.settimeout(0.005 if client_llm else 0.5)
            while not done() and not stop_reading:
                drain_client_llm()
                if time.monotonic() > deadline:
                    if mark_error:
                        result.error = result.error or "timeout"
                    return
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as exc:
                    result.error = f"transport failed: {type(exc).__name__}"
                    stop_reading = True
                    return
                if msg is None or msg == "":
                    result.error = result.error or "server closed"
                    stop_reading = True
                    return
                handle(msg)
                drain_client_llm()

        # Turn-gated pacing is required when the SERVER owns the LLM loop: it
        # answers and speaks after every mid-session asr_final, and streaming the
        # next utterance into that makes the audio look like barge-in and slips
        # the final<->utterance pairing. The gate waits for the turn's own
        # per-turn tts_done (which the sequencer emits with
        # session_complete=false and then re-arms — voxedge tts_sequencer.py:104).
        turn_gated = server_loop and multi_utterance

        for i in range(multi_count):
            utt_idx_send = i
            send_audio(wav_pcm, mark_stop_for=i)
            if i < multi_count - 1:
                # Inject silence so VAD fires SPEECH_END between utterances.
                # No stop mark here — silence is not "user said stop".
                send_audio(silence_pcm, mark_stop_for=None)
                if turn_gated:
                    want_idx = i + 1
                    pump_until(lambda: utt_idx_recv >= want_idx or stop_reading,
                               timeout)
                    tts_done = False
                    pump_until(lambda: tts_done or stop_reading, timeout,
                               mark_error=False)
                if stop_reading:
                    break

        t_session_stop = t_stops[-1]

        if prepare_lead_ms > 0:
            ws.send(json.dumps({"type": "asr_prepare"}))
            deadline = time.monotonic() + (prepare_lead_ms / 1000.0)
            while time.monotonic() < deadline:
                drain_nonblocking()
                time.sleep(min(chunk_dur_s, 0.02))

        # In persistent multi-utterance mode this finalizes the current turn
        # but does not close the session; only single mode marks the session
        # closing in the server dispatcher.
        ws.send(json.dumps({"type": "asr_eos"}))
        if tts_enabled:
            # Wait for asr_final, then echo it to TTS to measure round-trip.
            pass  # handled in reader

        # Drain everything until session_complete (multi) or final (single)
        # plus optional TTS audio.
        if turn_gated:
            # Each gated turn consumed its own tts_done, so re-arm for the
            # closing response rather than treating it as already finished.
            tts_done = not tts_enabled
        if client_llm:
            pump_until(lambda: client_turn_final and tts_done and client_llm_done,
                       timeout)
        else:
            pump_until(lambda: got_session_complete and tts_done, timeout)

        # Truncate to actually-received utterances (the trailing
        # session_complete=true final lands on the last pending slot).
        result.utterances = utt_pending[:max(1, utt_idx_recv + 1)]
        if client_llm and not result.error:
            if not client_llm_called:
                result.error = "client LLM was not called"
            elif not result.assistant_text:
                result.error = "client LLM produced no assistant text"
            elif not result.tts_complete:
                result.error = "missing tts_done"
            elif result.tts_pcm_bytes <= 0:
                result.error = "missing TTS PCM"
    except Exception as exc:
        # Keep direct callers and JSON output free of provider/transport text.
        result.error = f"transport failed: {type(exc).__name__}"
    finally:
        client_llm_ctl["cancel_requested"] = True
        loop = client_llm_ctl.get("loop")
        task = client_llm_ctl.get("task")
        if loop is not None and task is not None and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        if client_llm_thread is not None:
            client_llm_thread.join(timeout=0.5)
        try: ws.close()
        except Exception: pass

    return result


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def print_summary(records: list[RunResult], tts_enabled: bool):
    metrics: dict[str, list[float]] = {
        "tfd_ms": [],
        "stop_to_endpoint_ms": [],
        "stop_to_final_ms": [],
    }
    if tts_enabled:
        metrics["final_to_tts_started_ms"] = []
        metrics["tts_started_to_audio_ms"] = []
        metrics["final_to_tts_audio_ms"] = []
        metrics["stop_to_tts_audio_ms"] = []

    for r in records:
        if r.error:
            continue
        for u in r.utterances:
            if u.tfd_ms is not None:                metrics["tfd_ms"].append(u.tfd_ms)
            if u.stop_to_endpoint_ms is not None:   metrics["stop_to_endpoint_ms"].append(u.stop_to_endpoint_ms)
            if u.stop_to_final_ms is not None:      metrics["stop_to_final_ms"].append(u.stop_to_final_ms)
        if tts_enabled:
            if r.final_to_tts_started_ms is not None: metrics["final_to_tts_started_ms"].append(r.final_to_tts_started_ms)
            if r.tts_started_to_audio_ms is not None: metrics["tts_started_to_audio_ms"].append(r.tts_started_to_audio_ms)
            if r.final_to_tts_audio_ms is not None: metrics["final_to_tts_audio_ms"].append(r.final_to_tts_audio_ms)
            if r.stop_to_tts_audio_ms is not None:  metrics["stop_to_tts_audio_ms"].append(r.stop_to_tts_audio_ms)

    print("\n=== /v2v/stream latency summary ===", file=sys.stderr)
    print(f"{'metric':<26} {'n':>3} {'mean':>8} {'p50':>8} {'min':>8} {'max':>8}",
          file=sys.stderr)
    for k, vs in metrics.items():
        s = summarize(vs)
        if not s:
            print(f"{k:<26}  -- no samples --", file=sys.stderr)
            continue
        print(f"{k:<26} {s['n']:>3} {s['mean']:>7.0f}ms {s['p50']:>7.0f}ms "
              f"{s['min']:>7.0f}ms {s['max']:>7.0f}ms", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="host:port (no scheme)")
    ap.add_argument("--wav", required=True, help="WAV file (mono, any sample rate)")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--vad", choices=["silero", "none"], default="silero",
                    help="server-side VAD backend (default: silero; none uses client EOS)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="discard first N runs")
    ap.add_argument("--tts", action="store_true", help="enable TTS round-trip")
    ap.add_argument("--tts-language", default="auto")
    ap.add_argument("--multi", type=int, default=1,
                    help="utterances per session (>1 enables multi_utterance)")
    ap.add_argument("--silence-ms", type=int, default=700,
                    help="silence injected between utterances (multi mode)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="send audio as fast as possible (default: real-time pacing)")
    ap.add_argument("--prepare-lead-ms", type=int, default=0,
                    help="send asr_prepare after speech and wait this long before asr_eos")
    ap.add_argument("--server-loop", action="store_true",
                    help="server runs the LLM itself (OVS_V2V_SERVER_LOOP=1): do NOT "
                         "echo asr_final back as text; measure the server's own reply. "
                         "Implies --tts for measurement purposes.")
    ap.add_argument("--client-llm", action="store_true",
                    help="run one client-owned LLM turn after ASR final; "
                         "requires single-turn client-loop mode")
    args = ap.parse_args()
    if args.client_llm and (args.server_loop or args.multi != 1):
        ap.error("--client-llm is mutually exclusive with --server-loop and requires --multi 1")
    if args.server_loop:
        args.tts = True
    if args.client_llm:
        args.tts = True
    wav_pcm, dur = load_wav_16k_i16(args.wav)
    print(f"wav: {args.wav}  dur={dur:.2f}s  size={len(wav_pcm)} bytes  "
          f"multi={args.multi}  tts={args.tts}", file=sys.stderr)

    total = args.runs + args.warmup
    records: list[RunResult] = []
    for i in range(total):
        tag = "warmup" if i < args.warmup else "run"
        try:
            r = run_once(
                args.host, wav_pcm, dur,
                language=args.language,
                vad=args.vad,
                tts_enabled=args.tts,
                tts_language=args.tts_language,
                multi_count=args.multi,
                silence_ms=args.silence_ms,
                realtime=not args.no_realtime,
                prepare_lead_ms=args.prepare_lead_ms,
                server_loop=args.server_loop,
                client_llm=args.client_llm,
            )
        except Exception as e:
            message = type(e).__name__ if args.client_llm else f"{type(e).__name__}: {e}"
            r = RunResult(audio_dur_s=dur, error=message)
        print(f"[{tag} {i+1}/{total}] err={r.error or '-'} "
              f"utts={len(r.utterances)} "
              f"first_final={r.utterances[0].stop_to_final_ms if r.utterances else None}",
              file=sys.stderr)
        # Let the previous session's limiter slot be released before the next
        # connect, otherwise a run can be rejected mid-suite.
        time.sleep(1.5)
        if i >= args.warmup:
            records.append(r)
        # Emit per-run JSON to stdout
        sys.stdout.write(json.dumps(asdict(r)) + "\n")
        sys.stdout.flush()

    print_summary(records, args.tts)
    if args.client_llm and any(r.error for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
