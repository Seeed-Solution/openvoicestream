"""CPU fake-protocol tests for the app ASR performance gate.

These tests are in-memory and never touch a device, GPU, native worker, or a
real service. They pin the protocol and metric semantics that the driver relies
on so a real benchmark cannot silently measure the wrong thing:

  * a partial emitted DURING upload is timestamped from the first PCM send;
  * absence of a nonempty partial is UNPROVEN, never a fabricated number;
  * a duplicate final or an error/abnormal close fails the row;
  * a stalled server is bounded by the shared deadline and the receiver is
    cleaned up;
  * percentile outlier/warm inclusion matches the frozen 0-based rule;
  * a bad manifest hash is rejected;
  * failed rows are retained with zero exclusion;
  * the comparator refuses a service-identity mismatch.

Run: ``pytest server/tests/test_edgellm_asr_ws_perf_gate.py``
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

_DRIVER_PATH = Path(__file__).resolve().parents[2] / "bench" / "perf" / "edgellm_asr_ws_perf_gate.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("edgellm_asr_ws_perf_gate", _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ through
    # sys.modules and raises AttributeError when the module is not present.
    sys.modules["edgellm_asr_ws_perf_gate"] = module
    spec.loader.exec_module(module)
    return module


gate = _load_driver()


# ──────────────────────────────────────────────────────────────────────
# In-memory fake websocket
# ──────────────────────────────────────────────────────────────────────


class FakeServerConn:
    """A deterministic fake of the /asr/stream server.

    ``script`` is a callable invoked as ``script(conn)`` at the moment the
    first PCM byte is sent. It may emit frames at chosen points relative to
    ``sent_bytes`` to model a partial that arrives DURING upload.
    """

    def __init__(self, script, *, chunk_bytes_hint: int = 0) -> None:
        self._script = script
        self._inbox: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = asyncio.Event()
        self._close_code: Optional[int] = None
        self.sent_bytes = 0
        self.send_calls = 0
        self._script_task: Optional[asyncio.Task] = None
        self.primitive_emitted = False

    async def send(self, data: Any) -> None:
        self.send_calls += 1
        if isinstance(data, (bytes, bytearray)) and data:
            self.sent_bytes += len(data)
        if self.send_calls == 1:
            # Start the script as soon as the first send lands; the receiver
            # task already exists by construction in run_ws_utterance.
            self._script_task = asyncio.create_task(self._script(self))
            await asyncio.sleep(0)  # let it emit any immediate frames

    async def recv(self) -> Any:
        if self._closed.is_set() and self._inbox.empty():
            raise gate.ConnClosed(code=self._close_code, reason="fake close")
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=0.05)
        except asyncio.TimeoutError:
            if self._closed.is_set():
                raise gate.ConnClosed(code=self._close_code, reason="fake close")
            # Re-raise as a retryable timeout via a nested loop.
            return await self.recv()

    async def close(self, code: int = 1000) -> None:
        self._closed.set()
        await self._cancel_script()

    async def _cancel_script(self) -> None:
        task = self._script_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # Server-side helpers ------------------------------------------------

    async def emit_json(self, payload: dict[str, Any]) -> None:
        await self._inbox.put(json.dumps(payload))

    async def emit_bytes(self, payload: bytes) -> None:
        self.primitive_emitted = True
        await self._inbox.put(payload)

    async def close_server(self, code: int = 1000) -> None:
        self._close_code = code
        self._closed.set()


def _collector():
    """Return (factory, list) capturing created connections."""
    conns: list[FakeServerConn] = []

    def factory_maker(script):
        async def factory():
            conn = FakeServerConn(script)
            conns.append(conn)
            return conn

        return factory

    return factory_maker, conns


PCM = b"\x00\x01" * 16000  # 1 second of PCM16


def _run(coro):
    return asyncio.run(coro)


def _call_run_ws(script, pcm=PCM, **overrides):
    return _run(_call_run_ws_async(script, pcm=pcm, **overrides))


async def _call_run_ws_async(script, pcm=PCM, **overrides):
    factory_maker, conns = _collector()
    factory = factory_maker(script)
    params = dict(
        order=0,
        file="x.wav",
        item_id="x",
        warm=False,
        pair_index=None,
        audio_s=len(pcm) / 2 / 16000,
        transcript="hello world",
        chunk_bytes=3200,
        pace_s=0.0,
        request_deadline_s=2.0,
        post_final_window_s=0.05,
    )
    params.update(overrides)
    result = await gate.run_ws_utterance(factory, pcm, **params)
    return result, conns


# ──────────────────────────────────────────────────────────────────────
# 1. Early partial during upload
# ──────────────────────────────────────────────────────────────────────


def test_early_partial_during_upload_is_timestamped_from_first_send():
    async def script(conn: FakeServerConn) -> None:
        # Emit a partial after roughly the second chunk, i.e. DURING upload.
        while conn.sent_bytes < 6400:
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "partial", "text": "hel", "is_final": False})
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "final", "text": "hello world", "is_final": True})
        await conn.close_server(1000)

    result, _ = _call_run_ws(script)
    assert result.ok, result.error
    assert result.first_partial_arrival_mono is not None
    assert result.first_partial_latency_s is not None
    # Keyed off the FIRST SEND, so it must be strictly less than the total
    # request wall (the final arrives after upload completes).
    assert result.first_partial_latency_s < result.request_wall_s
    assert result.final_count == 1


# ──────────────────────────────────────────────────────────────────────
# 2. Absent partial → UNPROVEN, not a fabricated zero
# ──────────────────────────────────────────────────────────────────────


def test_absent_partial_is_unproven():
    async def script(conn: FakeServerConn) -> None:
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "final", "text": "hi", "is_final": True})
        await conn.close_server(1000)

    result, _ = _call_run_ws(script)
    assert result.ok
    assert result.first_partial_latency_s is None

    corpus = gate.Corpus(
        manifest_sha256=gate.FROZEN_MANIFEST_SHA256,
        manifest_path="m",
        corpus_dir_declared="",
        items=[],
    )
    metrics = gate.compute_metrics([result], [], corpus, identity_present=True)
    assert metrics["b1"]["first_partial_latency_s"] is None
    assert metrics["b1"]["first_partial_unproven"] is True
    assert metrics["b1"]["first_partial_coverage"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# 3. Duplicate final / error close / primitive frame fail positives
# ──────────────────────────────────────────────────────────────────────


def test_duplicate_final_fails_row():
    async def script(conn: FakeServerConn) -> None:
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "final", "text": "a", "is_final": True})
        await conn.emit_json({"type": "final", "text": "a", "is_final": True})
        await conn.close_server(1000)

    result, _ = _call_run_ws(script)
    assert not result.ok
    assert "duplicate" in (result.error or "") or "exactly one final" in (result.error or "")
    assert result.final_count == 2


def test_error_frame_fails_row():
    async def script(conn: FakeServerConn) -> None:
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "error", "error": "boom", "is_final": True})
        await conn.close_server(1011)

    result, _ = _call_run_ws(script)
    assert not result.ok
    assert "error" in (result.error or "").lower()


def test_abnormal_close_fails_row():
    async def script(conn: FakeServerConn) -> None:
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "final", "text": "a", "is_final": True})
        await conn.close_server(1011)

    result, _ = _call_run_ws(script)
    assert not result.ok
    assert "abnormal close" in (result.error or "")


def test_primitive_binary_frame_fails_row():
    async def script(conn: FakeServerConn) -> None:
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_bytes(b"\x00\x01")
        await conn.emit_json({"type": "final", "text": "a", "is_final": True})
        await conn.close_server(1000)

    result, _ = _call_run_ws(script)
    assert not result.ok
    assert "primitive" in (result.error or "")


# ──────────────────────────────────────────────────────────────────────
# 4. Stall timeout and receiver cleanup
# ──────────────────────────────────────────────────────────────────────


def test_stall_is_bounded_and_cleaned_up():
    async def script(conn: FakeServerConn) -> None:
        # Never emit anything; the request deadline must bound the wait.
        while not conn._closed.is_set():
            await asyncio.sleep(0.01)

    async def scenario():
        result, conns = await _call_run_ws_async(script, request_deadline_s=0.2)
        assert not result.ok
        assert "deadline" in (result.error or "")
        # The driver must have called close() on its own socket.
        assert conns and conns[0]._closed.is_set()
        # No lingering asyncio tasks from the receiver.
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        assert not pending

    _run(scenario())


def test_receiver_cleanup_leaves_no_pending_tasks():
    async def script(conn: FakeServerConn) -> None:
        while conn.sent_bytes < len(PCM):
            await asyncio.sleep(0.001)
        await conn.emit_json({"type": "final", "text": "ok", "is_final": True})
        await conn.close_server(1000)

    async def scenario():
        await _call_run_ws_async(script)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        assert not pending

    _run(scenario())


# ──────────────────────────────────────────────────────────────────────
# 5. Percentile outlier / warm inclusion (frozen 0-based rule)
# ──────────────────────────────────────────────────────────────────────


def test_percentile_is_frozen_zero_based_rule():
    values = list(range(1, 101))  # 1..100
    # ceil(0.95 * 100) = 95 -> sorted[95] == 96 under the 0-based rule.
    assert gate.percentile(values, 0.95) == 96
    # The old subtract-1 gate returned sorted[94] == 95; assert we differ.
    assert gate.percentile(values, 0.95) != 95
    # P50: ceil(0.5*100)=50 -> sorted[50] == 51.
    assert gate.percentile(values, 0.50) == 51
    # Small n edge: min(n-1, ...) must clamp.
    assert gate.percentile([7.0], 0.95) == 7.0
    assert gate.percentile([1.0, 2.0], 0.95) == 2.0


def test_warm_and_outlier_rows_are_included_in_stats():
    rows = [
        gate.UtteranceResult(
            order=i, file=f"{i}.wav", id=f"i{i}", warm=i < 3, pair_index=None,
            ok=True, audio_s=1.0, request_wall_s=1.0 + i, transcript="x",
        )
        for i in range(100)
    ]
    corpus = gate.Corpus(
        manifest_sha256=gate.FROZEN_MANIFEST_SHA256,
        manifest_path="m",
        corpus_dir_declared="",
        items=[],
    )
    metrics = gate.compute_metrics(rows, [], corpus, identity_present=True)
    assert metrics["b1"]["rows_total"] == 100
    assert metrics["b1"]["rows_ok"] == 100
    # Warm rows (order 0,1,2) are present: max latency includes order 99.
    assert metrics["b1"]["request_wall_s"]["max"] == 100.0
    assert metrics["b1"]["request_wall_s"]["count"] == 100


# ──────────────────────────────────────────────────────────────────────
# 6. Bad manifest / hash
# ──────────────────────────────────────────────────────────────────────


def test_bad_manifest_sha_is_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"corpus_dir": str(tmp_path), "items": []}), encoding="utf-8"
    )
    with pytest.raises(gate.ManifestError):
        gate.load_corpus(manifest, "deadbeef" * 8)


def test_good_manifest_loads_with_corpus_root_override(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpus_dir": "/nonexistent/declared",
                "items": [
                    {
                        "file": "a.wav",
                        "sha256": "aa" * 32,
                        "rate": 16000,
                        "channels": 1,
                        "bit_depth": 16,
                        "duration_s": 1.0,
                        "transcript": "HELLO",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sha = gate.sha256_file(manifest)
    root = tmp_path / "root"
    corpus = gate.load_corpus(manifest, sha, corpus_root=root)
    assert corpus.items[0].path == str(root / "a.wav")
    assert corpus.ref_words == 1


def _write_long_diag_fixture(tmp_path: Path, *, transcript: str = "", duration_s: float = 90.0):
    import hashlib
    import wave

    wav = tmp_path / "long.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * (16000 * 90))
    manifest = tmp_path / "long-manifest.json"
    manifest.write_text(json.dumps({
        "corpus_dir": str(tmp_path),
        "items": [{
            "file": wav.name,
            "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "rate": 16000, "channels": 1, "bit_depth": 16,
            "duration_s": duration_s, "transcript": transcript,
            "lang": "en", "id": "long",
        }],
    }), encoding="utf-8")
    return wav, manifest


def test_single_long_diagnostic_admits_one_empty_transcript(tmp_path: Path):
    _, manifest = _write_long_diag_fixture(tmp_path)
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    config.single_long_diagnostic = True
    loader = gate.OwnedInputLoader(config)
    loader.run()
    snapshot = loader.snapshot()
    assert snapshot["failures"] == []
    assert len(snapshot["admitted"]) == 1
    assert snapshot["corpus"].items[0].transcript == ""
    assert loader.cleanup_proven()


def test_expired_default_admission_reports_without_loader(tmp_path: Path):
    import time

    config = _dummy_config(tmp_path)
    report = asyncio.run(gate.admit_corpus_inputs(config, time.monotonic() - 1.0))
    assert report["status"] == "not_started"
    assert report["summary"] == "input admission not_started: 0/100 items admitted"
    assert report["pending"]["cleanup_complete"] is False


def test_expired_single_long_admission_reports_one_denominator(tmp_path: Path):
    import time

    config = _dummy_config(tmp_path)
    config.single_long_diagnostic = True
    report = asyncio.run(gate.admit_corpus_inputs(config, time.monotonic() - 1.0))
    assert report["status"] == "not_started"
    assert report["summary"] == "input admission not_started: 0/1 items admitted"
    assert report["loader"] is None


def test_single_long_diagnostic_keeps_default_one_item_rejected(tmp_path: Path):
    _, manifest = _write_long_diag_fixture(tmp_path)
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    loader = gate.OwnedInputLoader(config)
    loader.run()
    assert loader.snapshot()["failures"]
    assert "item count" in loader.snapshot()["failures"][0]["error"]


def test_single_long_diagnostic_rejects_wrong_item_sha(tmp_path: Path):
    _, manifest = _write_long_diag_fixture(tmp_path)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["items"][0]["sha256"] = "00" * 32
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    config.single_long_diagnostic = True
    loader = gate.OwnedInputLoader(config)
    loader.run()
    assert "sha256 mismatch" in loader.snapshot()["failures"][0]["error"]


def test_single_long_diagnostic_rejects_wrong_pcm_metadata(tmp_path: Path):
    _, manifest = _write_long_diag_fixture(tmp_path)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["items"][0]["channels"] = 2
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    config.single_long_diagnostic = True
    loader = gate.OwnedInputLoader(config)
    loader.run()
    assert "manifest channels=2" in loader.snapshot()["failures"][0]["error"]


@pytest.mark.parametrize(
    ("transcript", "duration_s", "detail"),
    [("spoken", 90.0, False), ("", 89.0, False), ("", 90.0, True)],
)
def test_single_long_diagnostic_rejects_quality_or_detail_inputs(
    tmp_path: Path, transcript: str, duration_s: float, detail: bool,
):
    _, manifest = _write_long_diag_fixture(
        tmp_path, transcript=transcript, duration_s=duration_s
    )
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    config.single_long_diagnostic = True
    if detail:
        config.corpus_detail = tmp_path / "detail.json"
        config.corpus_detail_sha256 = "11" * 32
    loader = gate.OwnedInputLoader(config)
    loader.run()
    assert loader.snapshot()["failures"]


def test_single_long_diagnostic_run_gate_uses_one_real_phase(monkeypatch, tmp_path: Path):
    _, manifest = _write_long_diag_fixture(tmp_path)
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    config.single_long_diagnostic = True

    async def fake_probe(*_args, **_kwargs):
        return {"identity_present": True}

    async def fake_run_one(*_args, **_kwargs):
        return gate.UtteranceResult(
            order=0, file="long.wav", id="long", warm=False,
            pair_index=None, ok=True, audio_s=90.0, request_wall_s=90.0,
            transcript="diagnostic output",
        )

    monkeypatch.setattr(gate, "probe_service_identity", fake_probe)
    monkeypatch.setattr(gate, "_run_one", fake_run_one)
    outcome = asyncio.run(gate.run_gate(config))
    assert len(outcome.b1) == 1
    assert outcome.b1_phase.name == "single-long-diagnostic"
    document = gate.outcome_to_dict(outcome)
    assert document["diagnostic"]["status"] == "UNPROVEN"
    assert document["qualification"]["status"] == "UNPROVEN"


def test_explicit_zh_binds_ws_query_and_http_multipart_language(monkeypatch):
    async def exercise_ws():
        calls = []
        class FakeWs:
            protocol = None
            close_code = 1000
            async def close(self, code=1000): pass
        async def connect(url, **kwargs):
            calls.append(url)
            return FakeWs()
        fake = type("Websockets", (), {"connect": staticmethod(connect)})
        monkeypatch.setitem(sys.modules, "websockets", fake)
        factory = gate._make_ws_factory("http://127.0.0.1:9", "/asr/stream", "zh", 16000)
        await factory()
        assert calls == ["ws://127.0.0.1:9/asr/stream?language=zh&sample_rate=16000&vad=none"]
    asyncio.run(exercise_ws())
    body = gate.build_multipart(b"pcm", "a.wav", "b", language="zh")
    assert b'name="language"' in body
    assert b"\r\nzh\r\n" in body


def test_omitted_language_preserves_ws_en_and_http_field_omission(monkeypatch):
    async def exercise_ws():
        calls = []
        class FakeWs:
            protocol = None
            close_code = 1000
        async def connect(url, **kwargs):
            calls.append(url)
            return FakeWs()
        fake = type("Websockets", (), {"connect": staticmethod(connect)})
        monkeypatch.setitem(sys.modules, "websockets", fake)
        factory = gate._make_ws_factory("http://127.0.0.1:9", "/asr/stream", "en", 16000)
        await factory()
        assert calls == ["ws://127.0.0.1:9/asr/stream?language=en&sample_rate=16000&vad=none"]
    asyncio.run(exercise_ws())
    body = gate.build_multipart(b"pcm", "a.wav", "b")
    assert b'name="language"' not in body


def test_run_gate_propagates_requested_language_to_ws_factory(monkeypatch, tmp_path: Path):
    _, manifest = _write_long_diag_fixture(tmp_path)
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = gate.sha256_file(manifest)
    config.single_long_diagnostic = True
    seen = []

    async def fake_probe(*_args, **_kwargs):
        return {"identity_present": True}

    async def fake_run_one(*_args, **_kwargs):
        return gate.UtteranceResult(order=0, file="long.wav", id="long", warm=False,
                                    pair_index=None, ok=True, audio_s=90.0,
                                    request_wall_s=90.0, transcript="diagnostic output")

    def capture_factory(base_url, ws_path, language, sample_rate):
        seen.append(language)
        async def unused():
            raise AssertionError("fixture does not open a real socket")
        return unused

    monkeypatch.setattr(gate, "probe_service_identity", fake_probe)
    monkeypatch.setattr(gate, "_run_one", fake_run_one)
    monkeypatch.setattr(gate, "_make_ws_factory", capture_factory)
    config.language = "zh"
    asyncio.run(gate.run_gate(config))
    config.output = tmp_path / "out-default"
    config.language = None
    asyncio.run(gate.run_gate(config))
    assert seen == ["zh", "en"]


def test_wav_hash_mismatch_is_a_failed_row(tmp_path: Path, monkeypatch):
    import hashlib
    import wave

    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "corpus_dir": str(tmp_path),
        "items": [{
            "file": wav.name, "sha256": "00" * 32, "rate": 16000,
            "channels": 1, "bit_depth": 16, "duration_s": 0.1,
            "transcript": "x", "lang": "en", "id": "i",
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(gate, "FROZEN_ITEM_COUNT", 1)
    config = _dummy_config(tmp_path)
    config.manifest = manifest
    config.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    loader = gate.OwnedInputLoader(config)
    loader.run()
    snapshot = loader.snapshot()
    assert snapshot["failures"]
    assert "sha256 mismatch" in snapshot["failures"][0]["error"]


def _dummy_config(tmp_path: Path) -> "gate.RunConfig":
    return gate.RunConfig(
        base_url="http://127.0.0.1:1",
        mode="ws",
        concurrency=1,
        manifest=tmp_path / "m.json",
        manifest_sha256="x",
        corpus_root=None,
        output=tmp_path / "out",
        chunk_ms=100,
        pace=False,
        request_deadline_s=1.0,
        overall_deadline_s=1.0,
        post_final_window_s=0.05,
        label="t",
    )


# ──────────────────────────────────────────────────────────────────────
# 7. Failed-row retention (no exclusion)
# ──────────────────────────────────────────────────────────────────────


def test_failed_rows_are_retained_and_qualification_not_passes():
    rows = [
        gate.UtteranceResult(
            order=i, file=f"{i}.wav", id=f"i{i}", warm=i < 3, pair_index=None,
            ok=(i != 5), audio_s=1.0, request_wall_s=1.0, transcript="x",
            error=None if i != 5 else "boom",
        )
        for i in range(100)
    ]
    corpus = gate.Corpus(
        manifest_sha256=gate.FROZEN_MANIFEST_SHA256,
        manifest_path="m",
        corpus_dir_declared="",
        items=[],
    )
    metrics = gate.compute_metrics(rows, [], corpus, identity_present=True)
    assert metrics["b1"]["rows_total"] == 100
    assert metrics["b1"]["rows_ok"] == 99
    assert metrics["b1"]["rows_failed"] == 1
    qual = gate.qualification(rows, [], corpus, identity_present=True)
    # Empty corpus cannot be UNPROVEN on items here because the test corpus is
    # synthetic; the failed row must still force NOTQUALIFIED.
    assert qual["status"] == "NOTQUALIFIED"
    assert any("failed rows" in r for r in qual["reasons"])


def test_missing_identity_is_unproven():
    rows = [
        gate.UtteranceResult(
            order=i, file=f"{i}.wav", id=f"i{i}", warm=i < 3, pair_index=None,
            ok=True, audio_s=1.0, request_wall_s=1.0, transcript="x",
        )
        for i in range(100)
    ]
    corpus = gate.Corpus(
        manifest_sha256=gate.FROZEN_MANIFEST_SHA256,
        manifest_path="m",
        corpus_dir_declared="",
        items=[],
    )
    qual = gate.qualification(rows, [], corpus, identity_present=False)
    assert qual["status"] == "UNPROVEN"
    assert any("identity absent" in r for r in qual["reasons"])


# ──────────────────────────────────────────────────────────────────────
# 8. Comparator identity mismatch
# ──────────────────────────────────────────────────────────────────────


def _run_doc(model_id: str, backend: str, mode: str = "ws", pace: bool = True):
    return {
        "mode": mode,
        "config": {"pace": pace, "chunk_ms": 100},
        "service_identity": {"asr_model_id": model_id, "asr_backend": backend},
        "artifact_identity": {"identity": {
            "target_device": "test-device", "sdk_version": "sdk",
            "upstream_commit": "upstream", "worker_sha256": "worker",
            "plugin_sha256": "plugin", "profile_family": "profile",
            "base_profile_sha256": "base", "profile_sha256": "profile-sha",
            "engine_sha256": "engine", "config_sha256": "config",
            "slot_variant": "baseline",
        }},
        "corpus": {
            "manifest_sha256": gate.FROZEN_MANIFEST_SHA256,
            "items": [{"sha256": "11" * 32}, {"sha256": "22" * 32}],
        },
        "metrics": {
            "b1": {"request_wall_s": {"p50": 1.0, "p95": 2.0}, "rtf": {"p95": 0.1}},
            "b2": {
                "request_wall_s": {"p95": 3.0},
                "paired_throughput_audio_s_per_s": {"p50": 4.0},
            },
            "quality": {"wer": 0.03, "ref_words": 787},
        },
    }


def test_comparator_accepts_identical_identity():
    app = _run_doc("qwen3-asr", "trt-edgellm")
    base = _run_doc("qwen3-asr", "trt-edgellm")
    out = gate.compare_runs(app, base)
    assert out["problems"] == []
    assert out["identical_protocol"] is True
    assert out["ratios"]["b1_p95_request_wall"] == pytest.approx(1.0)


def test_comparator_rejects_service_identity_mismatch():
    app = _run_doc("qwen3-asr", "trt-edgellm")
    base = _run_doc("other-model", "trt-edgellm")
    out = gate.compare_runs(app, base)
    assert out["problems"]
    assert any("model_id mismatch" in p for p in out["problems"])


def test_comparator_rejects_corpus_hash_mismatch():
    app = _run_doc("qwen3-asr", "trt-edgellm")
    base = _run_doc("qwen3-asr", "trt-edgellm")
    base["corpus"]["items"] = [{"sha256": "99" * 32}, {"sha256": "22" * 32}]
    out = gate.compare_runs(app, base)
    assert any("ordered corpus hashes differ" in p for p in out["problems"])


def test_comparator_rejects_pacing_mismatch():
    app = _run_doc("qwen3-asr", "trt-edgellm", pace=True)
    base = _run_doc("qwen3-asr", "trt-edgellm", pace=False)
    out = gate.compare_runs(app, base)
    assert any("pacing mismatch" in p for p in out["problems"])


# ──────────────────────────────────────────────────────────────────────
# 9. Output-dir guard + no Popen in the driver source
# ──────────────────────────────────────────────────────────────────────


def test_output_dir_must_not_exist(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    config = _dummy_config(tmp_path)
    config.output = out
    config.manifest = tmp_path / "missing.json"
    with pytest.raises(SystemExit):
        _run(gate.run_gate(config))


def test_driver_never_uses_popen_or_signals():
    source = _DRIVER_PATH.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "os.kill" not in source
    assert "signal." not in source

# ──────────────────────────────────────────────────────────────────────
# 11. Phase-isolated B1/B2 quality contract
# ──────────────────────────────────────────────────────────────────────


def _quality_contract_corpus() -> Any:
    item = gate.CorpusItem(
        order=0, path="", file="a.wav", sha256="aa" * 32, rate=16000,
        channels=1, bit_depth=16, duration_s=1.0, transcript="alpha",
        lang="en", id="a",
    )
    return gate.Corpus(
        manifest_sha256=gate.FROZEN_MANIFEST_SHA256,
        manifest_path="m", corpus_dir_declared="", items=[item],
    )


def _quality_contract_row(text: str, *, ok: bool = True, pair_index: Optional[int] = None):
    return gate.UtteranceResult(
        order=0, file="a.wav", id="a", warm=False, pair_index=pair_index,
        ok=ok, audio_s=1.0, request_wall_s=1.0, transcript=text,
        error=None if ok else "fixture failure",
    )


def test_b1_only_quality_alias_and_phase_shape_are_compatible():
    metrics = gate.compute_metrics(
        [_quality_contract_row("alpha")], [], _quality_contract_corpus(),
        identity_present=True,
    )
    assert metrics["quality"] == metrics["quality_by_phase"]["b1"]
    assert metrics["quality"]["wer"] == 0.0
    assert metrics["quality_by_phase"]["b2"]["rows"] == []


def test_b2_only_quality_alias_and_wer_check_are_real():
    metrics = gate.compute_metrics(
        [], [_quality_contract_row("wrong", pair_index=0)],
        _quality_contract_corpus(), identity_present=True,
    )
    assert metrics["quality"] == metrics["quality_by_phase"]["b2"]
    assert metrics["quality"]["wer"] == 1.0
    thresholds = gate.evaluate_app_thresholds(
        {}, {}, metrics["quality"], quality_by_phase=metrics["quality_by_phase"]
    )
    assert thresholds["checks"]["b2_wer"]["status"] == "FAIL"


def test_b1_b2_quality_gate_cannot_hide_b2_regression():
    metrics = gate.compute_metrics(
        [_quality_contract_row("alpha")],
        [_quality_contract_row("wrong", pair_index=0)],
        _quality_contract_corpus(), identity_present=True,
    )
    thresholds = gate.evaluate_app_thresholds(
        {}, {}, metrics["quality"], quality_by_phase=metrics["quality_by_phase"]
    )
    assert thresholds["checks"]["b1_wer"]["status"] == "PASS"
    assert thresholds["checks"]["b2_wer"]["status"] == "FAIL"
    assert thresholds["status"] == "FAIL"


def test_b1_b2_quality_checks_both_pass_independently():
    metrics = gate.compute_metrics(
        [_quality_contract_row("alpha")],
        [_quality_contract_row("alpha", pair_index=0)],
        _quality_contract_corpus(), identity_present=True,
    )
    thresholds = gate.evaluate_app_thresholds(
        {}, {}, metrics["quality"], quality_by_phase=metrics["quality_by_phase"]
    )
    assert thresholds["checks"]["b1_wer"]["status"] == "PASS"
    assert thresholds["checks"]["b2_wer"]["status"] == "PASS"


def test_failed_b2_row_is_retained_in_phase_quality():
    metrics = gate.compute_metrics(
        [_quality_contract_row("alpha")],
        [_quality_contract_row("wrong", ok=False, pair_index=0)],
        _quality_contract_corpus(), identity_present=True,
    )
    b2_quality = metrics["quality_by_phase"]["b2"]
    assert b2_quality["rows"]
    assert b2_quality["rows"][0]["failed_row"] is True
    assert b2_quality["wer"] == 1.0


def test_paired_document_uses_b2_quality_for_quality_argument(monkeypatch):
    def paired_doc(wer):
        return {
            "mode": "ws",
            "config": {"pace": True, "chunk_ms": 100},
            "service_identity": {"asr_model_id": "q", "asr_backend": "b"},
            "corpus": {
                "manifest_sha256": gate.FROZEN_MANIFEST_SHA256,
                "items": [{"sha256": "11" * 32}, {"sha256": "22" * 32}],
            },
            "metrics": {
                "b1": {}, "b2": {}, "quality": {"wer": wer, "ref_words": 787},
            },
        }
    b1_doc = paired_doc(0.0)
    b2_doc = paired_doc(1.0)
    captured = {}
    monkeypatch.setattr(gate, "FROZEN_ITEM_COUNT", 2)
    monkeypatch.setattr(gate, "_protocol_identity_problems", lambda *a: [])
    monkeypatch.setattr(gate, "_shared_identity_problems", lambda *a: [])
    monkeypatch.setattr(gate, "_variant_identity_provenance", lambda *a: ([], {}))
    monkeypatch.setattr(gate, "_phase_provenance_problems", lambda *a: [])

    def fake_thresholds(b1_metrics, b2_metrics, quality, *, quality_by_phase=None):
        captured["quality"] = quality
        captured["quality_by_phase"] = quality_by_phase
        return {"status": "FAIL", "checks": {}}

    monkeypatch.setattr(gate, "evaluate_app_thresholds", fake_thresholds)
    gate.paired_threshold_document(b1_doc, b2_doc)
    assert captured["quality"] is b2_doc["metrics"]["quality"]
    assert captured["quality_by_phase"]["b1"]["wer"] == 0.0
    assert captured["quality_by_phase"]["b2"]["wer"] == 1.0


def _quality_contract_outcome(tmp_path: Path, *, b1, b2):
    return gate.RunOutcome(
        config=_dummy_config(tmp_path),
        corpus=_quality_contract_corpus(),
        pre={}, post={}, b1=b1, b2=b2,
        b1_phase=None, b2_phase=None, post_warm_repeat=None,
        source_hashes={}, started_wall="start", finished_wall="finish",
    )


def test_outcome_b2_only_populates_wer_but_stays_unproven(tmp_path: Path):
    outcome = _quality_contract_outcome(
        tmp_path, b1=[], b2=[_quality_contract_row("wrong", pair_index=0)]
    )
    document = gate.outcome_to_dict(outcome)
    assert document["app_thresholds"]["status"] == "UNPROVEN"
    assert document["app_thresholds"]["checks"]["b2_wer"]["status"] == "FAIL"


def test_outcome_b1_only_keeps_checks_unproven_none(tmp_path: Path):
    outcome = _quality_contract_outcome(
        tmp_path, b1=[_quality_contract_row("alpha")], b2=[]
    )
    document = gate.outcome_to_dict(outcome)
    assert document["app_thresholds"]["status"] == "UNPROVEN"
    assert document["app_thresholds"]["checks"] is None


def _resource_fixture(tmp_path: Path, *, start_ticks="77"):
    import hashlib

    csv_path = tmp_path / "resource.csv"
    sampler_path = tmp_path / "resource_sampler.py"
    sampler_path.write_text("# sealed sampler fixture\n")
    start = 50123456.0
    csv_path.write_text(
        "t,sample_start_mono,sample_end_mono,cpu_pct,mem_pct,accel_raw,"
        "ram_used_mib,ram_total_mib,gr3d_util_pct,cpu_temp_c,gpu_temp_c,"
        "thermal_max_c,temperatures_c,parse_status,missing_fields,pid,"
        "start_ticks,rss_bytes,process_status,sample_interval_s,"
        "accel_child_pid,accel_term_count,accel_child_status\n"
        f"{start},{start},{start + 0.01},1,2,,100,200,0,60,61,61,,OBSERVED_RAW,,12,{start_ticks},100,BOUND,0.01,,,\n"
        f"{start + 0.5},{start + 0.5},{start + 0.51},1,2,,100,200,0,60,61,61,,OBSERVED_RAW,,12,{start_ticks},120,BOUND,0.01,,,\n",
        encoding="utf-8",
    )
    binding = {
        "worker_host_pid": 12, "worker_container_pid": 30, "worker_start_ticks": 77,
        "worker_exe_sha256": "e" * 64, "worker_argv": ["worker"],
        "container_id": "f" * 64, "container_init_host_pid": 12,
        "container_init_start_ticks": 66, "boot_id": "boot-1",
    }
    sampler_sha = hashlib.sha256(sampler_path.read_bytes()).hexdigest()
    envelope = {
        "schema": 1, "kind": "asr-resource-evidence", "closure_sha256": "c" * 64,
        "phase": "asr.http.b1", "run_id": "run-1", "target": "orin-nx", "boot_id": "boot-1",
        "worker": binding,
        "sampler": {"argv": ["/usr/bin/python3", str(sampler_path), "--out", str(csv_path)],
                    "source_sha256": sampler_sha, "rc": 0,
                    "terminal": {"reaped": True, "remaining_group_members": []}},
        "workload_start_unix_ns": 1, "workload_end_unix_ns": 2,
        "workload_start_mono": start, "workload_end_mono": start + 0.5,
        "sample_start_mono": start, "sample_end_mono": start + 0.51,
        "csv": {"path": str(csv_path), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "size": csv_path.stat().st_size},
        "coverage": {"sample_count": 2, "max_gap_s": 0.49000000208616257, "valid_bound_count": 2,
                     "sample_start_mono": start, "sample_end_mono": start + 0.51},
        "cuda_counter_status": "UNPROVEN",
    }
    env_path = tmp_path / "resource.json"
    env_path.write_text(json.dumps(envelope), encoding="utf-8")
    return env_path, envelope


def test_sealed_resource_evidence_proves_rows_but_cuda_stays_unproven(tmp_path: Path):
    env_path, envelope = _resource_fixture(tmp_path)
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    result = gate._resource_evidence_report(
        env_path, expected
    )
    assert result["status"] == "PROVEN"
    assert result["cuda_status"] == "UNPROVEN"
    assert result["coverage"]["valid_bound_count"] == 2


def test_sealed_resource_evidence_rejects_pid_start_tick_drift(tmp_path: Path):
    env_path, envelope = _resource_fixture(tmp_path, start_ticks="88")
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    result = gate._resource_evidence_report(
        env_path, expected
    )
    assert result["status"] == "UNPROVEN"
    assert "binding" in result["reason"] or "stable" in result["reason"]


def test_sealed_resource_evidence_rejects_unreaped_sampler(tmp_path: Path):
    env_path, envelope = _resource_fixture(tmp_path)
    envelope["sampler"]["terminal"]["reaped"] = False
    env_path.write_text(json.dumps(envelope), encoding="utf-8")
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    result = gate._resource_evidence_report(env_path, expected)
    assert result["status"] == "UNPROVEN"
    assert "terminal" in result["reason"] or "reaped" in result["reason"]


def _rewrite_resource_fixture(env_path: Path, edit_csv):
    import hashlib

    env = json.loads(env_path.read_text(encoding="utf-8"))
    csv_path = Path(env["csv"]["path"])
    edit_csv(csv_path)
    env["csv"]["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    env["csv"]["size"] = csv_path.stat().st_size
    env_path.write_text(json.dumps(env), encoding="utf-8")
    return env


def test_sealed_resource_evidence_rejects_empty_and_nan_csv(tmp_path: Path):
    env_path, envelope = _resource_fixture(tmp_path)
    header = Path(envelope["csv"]["path"]).read_text().splitlines()[0] + "\n"
    _rewrite_resource_fixture(env_path, lambda p: p.write_text(header))
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"

    env_path, envelope = _resource_fixture(tmp_path)
    _rewrite_resource_fixture(
        env_path,
        lambda p: p.write_text(p.read_text().replace(",100,BOUND", ",nan,BOUND")),
    )
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"


def test_sealed_resource_evidence_rejects_missing_header_gap_and_coverage(tmp_path: Path):
    env_path, envelope = _resource_fixture(tmp_path)
    _rewrite_resource_fixture(
        env_path,
        lambda p: p.write_text(p.read_text().replace(",thermal_max_c,", ",thermal_missing,")),
    )
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"

    env_path, envelope = _resource_fixture(tmp_path)
    _rewrite_resource_fixture(
        env_path,
        lambda p: p.write_text(p.read_text().replace("50123456.5,50123456.5", "50123460.0,50123460.0")),
    )
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"

    env_path, envelope = _resource_fixture(tmp_path)
    env = json.loads(env_path.read_text())
    env["workload_end_mono"] = 50123460.0
    env_path.write_text(json.dumps(env))
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"


def test_sealed_resource_evidence_rejects_foreign_pid_and_sampler_source(tmp_path: Path):
    env_path, envelope = _resource_fixture(tmp_path)
    _rewrite_resource_fixture(
        env_path,
        lambda p: p.write_text(p.read_text().replace(",12,77,100,", ",99,77,100,")),
    )
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"

    env_path, envelope = _resource_fixture(tmp_path)
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": "wrong" * 16}
    assert gate._resource_evidence_report(env_path, expected)["status"] == "UNPROVEN"


def test_sealed_file_and_six_role_artifact_rejections(tmp_path: Path):
    p = tmp_path / "sealed.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        gate._sealed_regular(p, "0" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(p)
    with pytest.raises(ValueError, match="regular file"):
        gate._sealed_regular(link)

    validator, _ = gate._load_canonical_artifact_validator()
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"status": "OBSERVED", "target": "orin-nx", "records": []}))
    expected = [{"role": "worker", "target": "orin-nx", "path": "/x",
                 "sha256": "a" * 64, "size": 1, "identity": {"pin": "x"}}]
    result = validator.validate_artifact_observation(proof, expected, tmp_path)
    assert result["status"] == "HARD_FAIL"
    assert "record count" in result["reason"] or "transport" in result["reason"]


def test_offline_report_refuses_existing_output(tmp_path: Path):
    out = tmp_path / "report.json"
    out.write_text("old")
    args = type("Args", (), {"evidence_report": out})()
    with pytest.raises(SystemExit, match="already exists"):
        gate.offline_evidence_report(args)


def test_offline_phase_evidence_wrong_sha_is_rejected(tmp_path: Path):
    phase = tmp_path / "phase.json"
    phase.write_text("{}")
    args = type("Args", (), {
        "evidence_report": tmp_path / "report.json",
        "phase_evidence": phase,
        "phase_evidence_sha256": "0" * 64,
        "evidence_run": phase,
        "evidence_run_sha256": "0" * 64,
        "resource_evidence": phase,
        "resource_evidence_sha256": "0" * 64,
        "live_artifact_proof": phase,
        "live_artifact_proof_sha256": "0" * 64,
    })()
    assert gate.offline_evidence_report(args) == 0
    result = json.loads((tmp_path / "report.json").read_text())
    assert result["status"] == "UNPROVEN"
    assert "SHA mismatch" in result["reason"]

@pytest.mark.parametrize("drift", ["target", "worker", "sampler"])
def test_sealed_resource_evidence_rejects_phase_context_drift(tmp_path: Path, drift: str):
    env_path, envelope = _resource_fixture(tmp_path)
    expected = {"closure_sha256": envelope["closure_sha256"], "phase": envelope["phase"],
                "run_id": envelope["run_id"], "target": envelope["target"],
                "boot_id": envelope["boot_id"],
                "sampler_source_sha256": envelope["sampler"]["source_sha256"],
                "worker_binding": envelope["worker"]}
    if drift == "target":
        envelope["target"] = "foreign-target"
    elif drift == "worker":
        envelope["worker"] = dict(envelope["worker"], worker_host_pid=999)
    else:
        envelope["sampler"] = dict(envelope["sampler"], source_sha256="a" * 64)
    env_path.write_text(json.dumps(envelope), encoding="utf-8")
    result = gate._resource_evidence_report(env_path, expected)
    assert result["status"] == "UNPROVEN"
