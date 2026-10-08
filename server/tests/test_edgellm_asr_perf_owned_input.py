"""Owned-input boundary tests for bench/perf/edgellm_asr_ws_perf_gate.py.

Scope (root design /tmp/slv-v011-asr-perf-input-boundary-root-design.md):

  * ONE owned daemon input thread admits manifest + ordered WAVs; it is the
    sole owner/closer of its file wrappers.
  * A blocked input read must never stall the event loop; expiry returns an
    explicit pending report with NO probes/inference and NO cleanup claim.
  * Expired initial budget starts NO thread, NO read, NO probe.
  * Admitted frozen bytes are read ONCE and are the sole basis of SHA, PCM
    and HTTP payload; post-admission path mutation cannot affect them.
  * Malformed / hash / duration / cap / nonregular / count failures block ALL
    inference with the fixed frozen denominator and retained evidence.
  * Successful admission requires thread-finished + all owned wrappers closed.

These tests author controlled fakes with bounded teardown; they never leak
non-daemon threads and never touch a live service or model artifact.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional

import pytest

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2] / "bench" / "perf" / "edgellm_asr_ws_perf_gate.py"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location("edgellm_asr_ws_perf_gate_owned", _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ through
    # sys.modules and raises AttributeError when the module is not present.
    sys.modules["edgellm_asr_ws_perf_gate_owned"] = module
    spec.loader.exec_module(module)
    return module


gate = _load_driver()

FROZEN_COUNT = 100
PCM_ONE_FRAME_MS = b"\x00\x01"  # one 16-bit sample pair
DURATION_S = 0.01  # 160 samples at 16 kHz
PCM_BYTES = PCM_ONE_FRAME_MS * 160  # 320 bytes = 0.01 s


# ──────────────────────────────────────────────────────────────────────
# Controlled corpus / config builders
# ──────────────────────────────────────────────────────────────────────


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)


def _build_corpus(
    tmp_path: Path,
    *,
    count: int = FROZEN_COUNT,
    pcm: bytes = PCM_BYTES,
    duration_s: float = DURATION_S,
) -> tuple[Path, str]:
    """Write a controlled 100-item WAV corpus + manifest; return
    (manifest_path, actual_manifest_sha256)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(count):
        name = f"u{i:03d}.wav"
        p = corpus_dir / name
        _write_wav(p, pcm)
        items.append(
            {
                "file": name,
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "rate": 16000,
                "channels": 1,
                "bit_depth": 16,
                "duration_s": duration_s,
                "transcript": "hello world",
                "lang": "en",
                "id": f"i{i}",
            }
        )
    payload = json.dumps({"corpus_dir": str(corpus_dir), "items": items}).encode("utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(payload)
    return manifest, hashlib.sha256(payload).hexdigest()


def _config(
    tmp_path: Path,
    manifest: Path,
    sha: str,
    *,
    overall_deadline_s: float = 30.0,
) -> "gate.RunConfig":
    return gate.RunConfig(
        base_url="http://127.0.0.1:1",  # nothing here: any probe attempt fails
        mode="http",
        concurrency=1,
        manifest=manifest,
        manifest_sha256=sha,
        corpus_root=None,
        output=tmp_path / "out",
        chunk_ms=100,
        pace=False,
        request_deadline_s=1.0,
        overall_deadline_s=overall_deadline_s,
        post_final_window_s=0.05,
        label="owned-input",
    )


# ──────────────────────────────────────────────────────────────────────
# Fake opener / wrappers (ownership + blocking control)
# ──────────────────────────────────────────────────────────────────────


class RealWrapper:
    """Thin wrapper over a real buffered file; records closes (the owning
    thread must be the ONLY closer)."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self.close_calls = 0

    def read(self, n: int) -> bytes:
        return self._handle.read(n)

    def close(self) -> None:
        self.close_calls += 1
        self._handle.close()

    @property
    def closed(self) -> bool:
        # Actual proof delegates to the underlying buffered handle; the
        # fixture no longer hides the real closed state from the loader.
        return self._handle.closed

    def __repr__(self) -> str:  # pragma: no cover - evidence only
        return f"<RealWrapper closes={self.close_calls} closed={self._handle.closed}>"


class SlowWrapper:
    """Fake WAV wrapper whose first read blocks until released; proves the
    caller NEVER closes it and NEVER force-stops the blocked read."""

    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.read_started = threading.Event()
        self.close_calls = 0
        self.read_calls = 0
        self._closed = False

    def read(self, n: int) -> bytes:
        self.read_calls += 1
        self.read_started.set()
        # Bounded fake-side wait so test teardown can never hang forever.
        self.release.wait(timeout=30)
        return b""

    def close(self) -> None:
        self.close_calls += 1
        # Purely fake wrapper: it owns no real handle, so the honest closed
        # state is the transition performed by its OWN close().
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - evidence only
        return f"<SlowWrapper closes={self.close_calls} closed={self._closed}>"


class FakeOpener:
    """Manifests open as real (wrapped) files; WAVs open as the given wrapper
    factory. Counts opens so 'read exactly once' is observable."""

    def __init__(self, wav_wrapper_factory) -> None:
        self._wav_factory = wav_wrapper_factory
        self.opened_paths: list[str] = []
        self.wrappers: list[Any] = []

    def __call__(self, path: str):
        self.opened_paths.append(path)
        if path.endswith(".json"):
            wrapper = RealWrapper(open(path, "rb"))
        else:
            wrapper = self._wav_factory(path)
        self.wrappers.append(wrapper)
        return wrapper


def _join_all(loaders, timeout: float = 5.0) -> None:
    """Bounded own teardown: release nothing, just wait for exit."""
    deadline = time.monotonic() + timeout
    for loader in loaders:
        loader.join(timeout=max(0.0, deadline - time.monotonic()))
    for loader in loaders:
        assert not loader.is_alive(), "test leaked a live loader thread"


# ──────────────────────────────────────────────────────────────────────
# 1. Slow/hung reader: loop progress, deadline pending, sole close
# ──────────────────────────────────────────────────────────────────────


def test_slow_reader_blocks_input_not_event_loop_and_expiry_reports_pending(
    tmp_path: Path, monkeypatch
):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha, overall_deadline_s=0.5)
    release = threading.Event()
    opener = FakeOpener(lambda path: SlowWrapper(release))

    async def scenario():
        start = time.monotonic()
        admit_task = asyncio.create_task(
            gate.admit_corpus_inputs(config, start + 0.5, opener=opener)
        )
        # Unrelated async work must keep progressing while the input read is
        # blocked in the owned thread (event loop never waits on the thread).
        ticks = 0
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks += 1
        loop_elapsed = time.monotonic() - start
        assert not admit_task.done(), "admission finished while the read was blocked"
        report = await admit_task
        return ticks, loop_elapsed, report

    ticks, loop_elapsed, report = asyncio.run(scenario())
    assert ticks == 5
    assert loop_elapsed < 0.3, "event loop stalled behind the blocked input read"
    assert report["status"] == "pending"
    assert report["pending"]["thread_alive"] is True
    assert report["pending"]["cleanup_complete"] is False
    assert report["pending"]["open_wrappers"], "blocked wrapper must remain registered"
    assert "NOT canceled" in report["note"] and "force-stopped" in report["note"]
    blocked = [w for w in opener.wrappers if isinstance(w, SlowWrapper)]
    assert blocked and blocked[0].close_calls == 0, "no close while read is blocked"

    # Pending input blocks ALL probes/inference via run_gate (explicit
    # InputAdmissionError, never a silent empty success). The fake opener is
    # installed as the driver default so run_gate's OWN owned thread blocks
    # on the same controlled fake.
    monkeypatch.setattr(gate, "_default_owned_open", opener)
    with pytest.raises(gate.InputAdmissionError) as excinfo:
        asyncio.run(gate.run_gate(config))
    pending_report = excinfo.value.report
    assert pending_report["status"] == "pending"
    assert pending_report["loader"].is_alive()
    loaders = [report["loader"], pending_report["loader"]]

    # Release the read; the OWNED THREAD alone closes the wrapper.
    release.set()
    _join_all(loaders)
    for w in blocked:
        assert w.close_calls == 1, "sole close must come from the owning thread"
    for loader in loaders:
        assert not loader.open_wrappers()


# ──────────────────────────────────────────────────────────────────────
# 2. Expired initial budget: no helper thread, no read, no probe
# ──────────────────────────────────────────────────────────────────────


def test_expired_initial_budget_starts_no_thread_read_or_probe(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)
    threads_before = threading.active_count()

    expired = time.monotonic() - 1.0  # already expired absolute deadline
    report = asyncio.run(gate.admit_corpus_inputs(config, expired))
    assert report["status"] == "not_started"
    assert report["pending"]["thread_alive"] is False
    assert report["admitted_orders"] == []
    assert threading.active_count() == threads_before, "no helper thread may start"

    expired_config = _config(tmp_path, manifest, sha, overall_deadline_s=0.0)
    with pytest.raises(gate.InputAdmissionError) as excinfo:
        asyncio.run(gate.run_gate(expired_config))
    assert excinfo.value.report["status"] == "not_started"
    assert "NO helper thread" in excinfo.value.report["note"]
    assert threading.active_count() == threads_before


# ──────────────────────────────────────────────────────────────────────
# 3. Read once; frozen bytes feed SHA/PCM/HTTP; mutation immune; item_id
# ──────────────────────────────────────────────────────────────────────


class FakeHTTPWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, data: Any) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return


class FakeHTTPReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def read(self, n: int) -> bytes:
        await asyncio.sleep(0)
        out, self._payload = self._payload[:n], self._payload[n:]
        return out


def _fake_http_open_conn(store: list[FakeHTTPWriter]):
    body = json.dumps({"text": "hi"}).encode()
    head = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()

    async def opener(parsed: Any, deadline_mono: float):
        writer = FakeHTTPWriter()
        store.append(writer)
        return FakeHTTPReader(head + body), writer

    return opener


def test_admitted_frozen_bytes_read_once_and_immune_to_post_admission_mutation(
    tmp_path: Path,
):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)
    opener = FakeOpener(lambda path: RealWrapper(open(path, "rb")))

    async def admit():
        return await gate.admit_corpus_inputs(
            config, time.monotonic() + 30.0, opener=opener
        )

    report = asyncio.run(admit())
    assert report["status"] == "admitted", report["failures"]
    loader = report["loader"]
    assert report["pending"]["cleanup_complete"] is True
    assert not loader.is_alive()
    assert report["item_count"] == FROZEN_COUNT
    # Exactly ONE open per WAV plus ONE for the manifest: no repeated reads.
    wav_opens = [p for p in opener.opened_paths if p.endswith(".wav")]
    assert len(wav_opens) == FROZEN_COUNT
    assert len(opener.opened_paths) == FROZEN_COUNT + 1

    # Mutate the on-disk file AFTER admission: the frozen bytes must not change.
    victim = loader.corpus.items[42]
    original_bytes = loader.admitted[42].wav_bytes
    victim_path = Path(victim.path)
    victim_path.write_bytes(b"MUTATED-AFTER-ADMISSION" * 16)

    item = loader.corpus.items[42]
    corpus = loader.corpus
    writers: list[FakeHTTPWriter] = []

    async def one_row():
        return await gate._run_one(
            None, corpus, item, config, "http", 3200, 0.0,
            time.monotonic() + 10.0, warm=False, pair_index=None,
            http_open_conn=_fake_http_open_conn(writers),
            admitted=loader.admitted,
        )

    result = asyncio.run(one_row())
    assert result.ok, result.error
    # item_id contract: the row carries the manifest id (old driver passed
    # id=item.id against an item_id keyword — a TypeError for HTTP rows).
    assert result.id == "i42"
    body = bytes(writers[0].data)
    assert victim.file.encode() in body, "multipart filename from manifest"
    assert original_bytes in body, "HTTP payload must be the admitted frozen bytes"
    assert b"MUTATED-AFTER-ADMISSION" not in body, "no post-admission re-read"
    assert loader.admitted[42].check.sha256 == hashlib.sha256(original_bytes).hexdigest()

    _join_all([loader])


def test_path_only_caller_fails_closed_without_synchronous_reread(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path, count=2)
    payload = json.loads(manifest.read_bytes())
    item = gate.CorpusItem(
        order=0, path=payload["items"][0]["path"], file="u000.wav",
        sha256=payload["items"][0]["sha256"], rate=16000, channels=1,
        bit_depth=16, duration_s=DURATION_S, transcript="hello world",
        lang="en", id="i0",
    )
    corpus = gate.Corpus(sha, str(manifest), "", [item])
    config = _config(tmp_path, manifest, sha)
    before = Path(item.path).read_bytes()

    async def one_row():
        return await gate._run_one(
            None, corpus, item, config, "ws", 3200, 0.0,
            time.monotonic() + 5.0, warm=True, pair_index=None,
            admitted=None,  # no owned admission: legacy path-only caller
        )

    result = asyncio.run(one_row())
    assert not result.ok
    assert "owned input boundary" in (result.error or "")
    # Fail-closed migration: the file was NOT re-read behind the caller's
    # back and nothing claimed a successful input.
    assert Path(item.path).read_bytes() == before


# ──────────────────────────────────────────────────────────────────────
# 4. Malformed / hash / duration / cap / nonregular / count failures
# ──────────────────────────────────────────────────────────────────────


def _admit(config, opener=None):
    async def run():
        return await gate.admit_corpus_inputs(
            config, time.monotonic() + 30.0, opener=opener
        )
    return asyncio.run(run())


def _expect_failed_and_blocking(tmp_path: Path, manifest: Path, sha: str, needle: str):
    config = _config(tmp_path, manifest, sha)
    report = _admit(config)
    assert report["status"] == "failed", report
    assert report["failure_count"] >= 1
    assert report["admitted_orders"] != list(range(FROZEN_COUNT))
    assert any(needle in json.dumps(f) for f in report["failures"]), report["failures"]
    # Failed input blocks ALL probes/inference; fixed frozen denominator.
    with pytest.raises(gate.InputAdmissionError) as excinfo:
        asyncio.run(gate.run_gate(config))
    assert excinfo.value.report["status"] == "failed"
    assert excinfo.value.report["item_count"] in (None, FROZEN_COUNT)
    _join_all([excinfo.value.report["loader"], report["loader"]])


def test_hash_mismatch_blocks_inference(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    payload = json.loads(manifest.read_bytes())
    victim = Path(payload["items"][7]["path"])
    _write_wav(victim, b"\x07\x00" * 160)  # same shape, different content
    _expect_failed_and_blocking(tmp_path, manifest, sha, "sha256 mismatch")


def test_duration_mismatch_blocks_inference(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    payload = json.loads(manifest.read_bytes())
    # The declared duration must exceed the production DURATION_TOLERANCE_S
    # (0.01 s, strict greater-than): 0.02 s declared vs 0.01 s actual is a
    # 0.01 s diff, which equals the tolerance and is correctly ADMITTED;
    # 0.03 s gives a 0.02 s diff, strictly outside the tolerance.
    payload["items"][5]["duration_s"] = 0.03  # actual is 0.01 s
    data = json.dumps(payload).encode()
    manifest.write_bytes(data)
    _expect_failed_and_blocking(tmp_path, manifest, hashlib.sha256(data).hexdigest(), "duration mismatch")


def test_cap_exceeded_blocks_inference(tmp_path: Path):
    # Manifest declares 0.001 s items (tiny derived per-item cap of
    # 32 + 64 KiB bytes) but the WAV holds ~80 KB: the bounded read refuses
    # BEFORE unbounded allocation.
    manifest, sha = _build_corpus(tmp_path, pcm=b"\x00\x00" * 40000, duration_s=0.001)
    _expect_failed_and_blocking(tmp_path, manifest, sha, "cap exceeded")


def test_nonregular_input_blocks_inference(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    payload = json.loads(manifest.read_bytes())
    fifo = Path(payload["items"][3]["path"])
    fifo.unlink()
    os.mkfifo(fifo)
    _expect_failed_and_blocking(tmp_path, manifest, sha, "not a regular file")


def test_wrong_item_count_blocks_inference(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path, count=99)
    _expect_failed_and_blocking(tmp_path, manifest, sha, "item count")


def test_manifest_sha_mismatch_blocks_inference(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    _expect_failed_and_blocking(tmp_path, manifest, "ff" * 32, "manifest sha256 mismatch")


# ──────────────────────────────────────────────────────────────────────
# 5. Successful admission requires complete owned cleanup
# ──────────────────────────────────────────────────────────────────────


def test_successful_admission_requires_thread_finish_and_full_wrapper_close(
    tmp_path: Path,
):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)
    opener = FakeOpener(lambda path: RealWrapper(open(path, "rb")))
    report = _admit(config, opener=opener)
    assert report["status"] == "admitted", report["failures"]
    loader = report["loader"]
    assert report["input_duration_s"] is not None
    # Separate input duration is reported, and the caller's overall budget
    # (30 s here) demonstrably included it (admission happened inside it).
    assert report["input_duration_s"] > 0
    assert sorted(report["admitted_orders"]) == list(range(FROZEN_COUNT))
    assert report["ref_words"] == FROZEN_COUNT * 2  # "hello world" x 100
    assert report["totals"]["admitted_pcm_bytes"] == FROZEN_COUNT * len(PCM_BYTES)
    # Cleanup proof: thread finished AND every owned wrapper closed exactly
    # once by the owning thread.
    assert not loader.is_alive()
    assert not loader.open_wrappers()
    assert report["pending"]["cleanup_complete"] is True
    for w in opener.wrappers:
        assert w.close_calls == 1
    _join_all([loader])


def test_input_caps_are_derived_and_explicit(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)
    report = _admit(config)
    caps = report["caps"]
    assert caps is not None
    # Derived from actual manifest metadata: 0.01 s * 32000 B/s + 64 KiB margin.
    assert caps["per_item_max_bytes"] == 0.01 * 32000 + gate.INPUT_WAV_HEADER_MARGIN_BYTES
    assert (
        caps["aggregate_max_bytes"]
        == FROZEN_COUNT * (0.01 * 32000 + gate.INPUT_WAV_HEADER_MARGIN_BYTES)
    )
    assert caps["derived_expected_pcm_bytes"] == FROZEN_COUNT * 320
    assert "256 MiB" in caps["rationale"] and "1 GiB" in caps["rationale"]
    _join_all([report["loader"]])


# ──────────────────────────────────────────────────────────────────────
# 6. v2 repairs: close-failure ownership, cleanup-proven admission,
#    truncated-PCM frame-length proof, synchronized snapshot
# ──────────────────────────────────────────────────────────────────────


class BrokenCloseWrapper:
    """Real reads; close() RAISES and .closed stays False."""

    def __init__(self, path: str) -> None:
        self._handle = open(path, "rb")
        self.closed = False
        self.close_calls = 0

    def read(self, n: int) -> bytes:
        return self._handle.read(n)

    def close(self) -> None:
        self.close_calls += 1
        raise OSError("simulated close failure")

    def __repr__(self) -> str:  # pragma: no cover - evidence only
        return f"<BrokenCloseWrapper calls={self.close_calls} closed={self.closed}>"


class UnprovenCloseWrapper:
    """Real reads; close() returns but .closed stays False (unproven)."""

    def __init__(self, path: str) -> None:
        self._handle = open(path, "rb")
        self.closed = False
        self.close_calls = 0

    def read(self, n: int) -> bytes:
        return self._handle.read(n)

    def close(self) -> None:
        self.close_calls += 1
        self._handle.close()
        self.closed = False  # close outcome NOT proven

    def __repr__(self) -> str:  # pragma: no cover - evidence only
        return f"<UnprovenCloseWrapper calls={self.close_calls} closed={self.closed}>"


class MissingClosedWrapper:
    """Real reads; close() succeeds and closes the handle, but the wrapper
    exposes NO ``.closed`` attribute at all. Missing proof is UNKNOWN, never
    a default True, so this MUST remain dirty ownership."""

    def __init__(self, path: str) -> None:
        self._handle = open(path, "rb")
        self.close_calls = 0

    def read(self, n: int) -> bytes:
        return self._handle.read(n)

    def close(self) -> None:
        self.close_calls += 1
        self._handle.close()
        # Deliberately NO ``self.closed`` assignment/property.

    def __repr__(self) -> str:  # pragma: no cover - evidence only
        return f"<MissingClosedWrapper calls={self.close_calls} closed=<absent>>"


class RaisingClosedWrapper:
    """Real reads; close() succeeds but reading ``.closed`` RAISES. A failed
    proof getter is dirty ownership; the wrapper must stay registered."""

    def __init__(self, path: str) -> None:
        self._handle = open(path, "rb")
        self.close_calls = 0

    def read(self, n: int) -> bytes:
        return self._handle.read(n)

    @property
    def closed(self) -> bool:
        raise RuntimeError("closed proof getter failed")

    def close(self) -> None:
        self.close_calls += 1
        self._handle.close()

    def __repr__(self) -> str:  # pragma: no cover - evidence only
        return f"<RaisingClosedWrapper calls={self.close_calls} closed=<raises>>"


def _expect_dirty_close_blocks(wrapper_cls, needle: str, tmp_path: Path, monkeypatch):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)
    made: list[Any] = []

    def factory(path: str):
        wrapper = (
            RealWrapper(open(path, "rb")) if path.endswith(".json")
            else wrapper_cls(path)
        )
        made.append(wrapper)
        return wrapper

    report = _admit(config, opener=factory)
    assert report["status"] == "failed", report
    loader = report["loader"]
    dirty = [w for w in made if isinstance(w, wrapper_cls)]
    assert dirty, "dirty wrapper must have been created"
    # Ownership retained: the ACTUAL dirty object stays registered even after
    # the thread is dead; no retry duplicate close happened.
    assert not loader.is_alive()
    assert any(w is dirty[0] for w in loader.open_wrappers())
    assert dirty[0].close_calls == 1
    pending = report["pending"]
    assert pending["cleanup_complete"] is False
    assert pending["close_errors"], "unproven close must be ledgered"
    assert any(needle in e for e in pending["close_errors"])
    assert pending["thread_alive"] is False  # dirty reported even if dead
    # run_gate independently rejects: no probe, no inference.
    monkeypatch.setattr(gate, "_default_owned_open", factory)
    with pytest.raises(gate.InputAdmissionError) as excinfo:
        asyncio.run(gate.run_gate(config))
    assert excinfo.value.report["pending"]["cleanup_complete"] is False
    assert excinfo.value.report["status"] in ("failed", "pending")
    _join_all([loader, excinfo.value.report["loader"]])


def test_close_raise_keeps_ownership_and_blocks_inference(tmp_path: Path, monkeypatch):
    _expect_dirty_close_blocks(BrokenCloseWrapper, "close error", tmp_path, monkeypatch)


def test_unproven_close_keeps_ownership_and_blocks_inference(tmp_path: Path, monkeypatch):
    _expect_dirty_close_blocks(UnprovenCloseWrapper, "close not proven", tmp_path, monkeypatch)


def test_missing_closed_attribute_is_unknown_not_clean(tmp_path: Path, monkeypatch):
    # close() RETURNED and the handle closed, but the wrapper exposes no
    # ``.closed`` attribute: unproven close must keep ownership and block.
    _expect_dirty_close_blocks(
        MissingClosedWrapper, "close not proven", tmp_path, monkeypatch
    )


def test_raising_closed_property_is_unproven_and_blocks(tmp_path: Path, monkeypatch):
    # Reading ``.closed`` raises: a failed proof getter is dirty ownership.
    _expect_dirty_close_blocks(
        RaisingClosedWrapper, "closed getter raised", tmp_path, monkeypatch
    )


def test_truncated_pcm_fails_despite_matching_file_sha(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    payload = json.loads(manifest.read_bytes())
    victim = Path(payload["items"][9]["path"])
    truncated = victim.read_bytes()[:-100]  # header still declares full frames
    victim.write_bytes(truncated)
    # Manifest SHA matches the ACTUAL (truncated) file: the SHA alone must
    # NOT admit the item — the frame-length proof is independent.
    items = payload["items"]
    items[9]["sha256"] = hashlib.sha256(truncated).hexdigest()
    data = json.dumps(payload).encode()
    manifest.write_bytes(data)
    _expect_failed_and_blocking(
        tmp_path, manifest, hashlib.sha256(data).hexdigest(), "truncated PCM"
    )


def test_done_event_set_while_helper_still_alive_returns_bounded_pending(
    tmp_path: Path,
):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha, overall_deadline_s=0.5)
    release = threading.Event()

    class SlowExitLoader(gate.OwnedInputLoader):
        def __init__(self, cfg, opener=None):
            super().__init__(cfg, opener=opener)

        def run(self):
            self.done_event.set()  # done BEFORE actual exit: not exit proof
            release.wait(timeout=10)  # bounded fake-side wait

    def factory(cfg, opener=None):
        return SlowExitLoader(cfg, opener=opener)

    async def scenario():
        start = time.monotonic()
        return await gate.admit_corpus_inputs(
            config, start + 0.5, loader_factory=factory
        )

    t0 = time.monotonic()
    report = asyncio.run(scenario())
    elapsed = time.monotonic() - t0
    assert report["status"] == "pending"
    assert report["pending"]["thread_alive"] is True
    assert report["pending"]["cleanup_complete"] is False
    assert elapsed < 2.0, "bounded exit poll must not wait unboundedly"
    loader = report["loader"]
    release.set()
    _join_all([loader])
    assert loader.cleanup_proven() is True  # observable completion AFTER exit


def test_unexpected_helper_exit_incomplete_corpus_fails(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)

    class GhostLoader(gate.OwnedInputLoader):
        def run(self):
            # Unexpected terminal state: exits with NO corpus, NO admitted
            # items and NO ledgered failure.
            self.finished_mono = time.monotonic()
            self.done_event.set()

    report = asyncio.run(
        gate.admit_corpus_inputs(
            config, time.monotonic() + 30.0,
            loader_factory=lambda cfg, opener=None: GhostLoader(cfg, opener=opener),
        )
    )
    assert report["status"] == "failed"
    assert any(
        f.get("stage") == "input_admission_terminal" for f in report["failures"]
    ), report["failures"]
    assert report["admitted_orders"] == []
    # The terminal failure IS ledgered exactly once and the corpus was never
    # parsed (no item count, no admitted items, nothing to admit/infer).
    # Input validity and cleanup are DISTINCT: the helper exited cleanly with
    # zero opened wrappers and zero close errors, so cleanup_complete is
    # truthfully True even though admission failed.
    assert report["failure_count"] == 1, report["failures"]
    assert report["item_count"] is None
    assert report["totals"]["admitted_wav_bytes"] == 0
    assert report["pending"]["thread_alive"] is False
    assert report["pending"]["open_wrappers"] == []
    assert report["pending"]["close_errors"] == []
    assert report["pending"]["cleanup_complete"] is True
    _join_all([report["loader"]])


def test_partly_spent_deadline_has_no_minimum_floor(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha, overall_deadline_s=0.15)
    release = threading.Event()
    opener = FakeOpener(lambda path: SlowWrapper(release))

    async def scenario():
        start = time.monotonic()
        report = await gate.admit_corpus_inputs(
            config, start + 0.15, opener=opener
        )
        return start, report

    t0 = time.monotonic()
    start, report = asyncio.run(scenario())
    elapsed = time.monotonic() - t0
    assert report["status"] == "pending"
    assert elapsed < 1.0, f"no minimum floor wait allowed, took {elapsed:.3f}s"
    assert elapsed >= 0.15 * 0.9  # budget was actually respected, not ignored
    release.set()
    _join_all([report["loader"]])


def test_pre_done_poll_sleep_is_capped_by_partly_spent_remaining(
    tmp_path: Path, monkeypatch
):
    """Injected sleep + clock: a POSITIVE remaining budget that is SMALLER
    than INPUT_POLL_INTERVAL_S must cap the pre-done poll sleep to that
    remaining value (never the full poll). After expiry NO further sleep or
    helper starts, and the post-done poll is also capped."""
    manifest, sha = _build_corpus(tmp_path)
    config = _config(tmp_path, manifest, sha)
    release = threading.Event()
    opener = FakeOpener(lambda path: SlowWrapper(release))

    poll = gate.INPUT_POLL_INTERVAL_S
    # Remaining budget is positive but strictly smaller than one poll.
    remaining = poll * 0.4
    clock = {"now": 1000.0}
    sleeps: list[float] = []
    expired_sleeps: list[float] = []

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return clock["now"]

    class FakeSleepLoader(gate.OwnedInputLoader):
        def run(self) -> None:
            # Block on the fake wrapper: the thread stays ALIVE (done_event
            # is set by the injected sleep, never by this body).
            release.wait(timeout=10)

    created: list[FakeSleepLoader] = []

    def factory(cfg, opener=None):
        loader = FakeSleepLoader(cfg, opener=opener)
        created.append(loader)
        return loader

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        if any(s > 0 for s in sleeps):
            # Demonstrate that a sleep is never requested once the ORIGINAL
            # remaining budget is exhausted.
            if clock["now"] >= 1000.0 + remaining:
                expired_sleeps.append(delay)
        clock["now"] += delay
        # One injected sleep is enough to set done_event; the helper thread
        # itself remains blocked (alive) so the post-done poll path runs.
        if len(sleeps) == 1:
            created[0].done_event.set()

    class FakeAsyncio:
        sleep = staticmethod(fake_sleep)

    monkeypatch.setattr(gate, "time", FakeTime)
    monkeypatch.setattr(gate, "asyncio", FakeAsyncio)

    report = asyncio.run(
        gate.admit_corpus_inputs(
            config, clock["now"] + remaining, opener=opener, loader_factory=factory
        )
    )
    release.set()
    _join_all(created)

    assert report["status"] == "pending", report
    # The FIRST (pre-done) sleep was capped to the smaller remaining budget,
    # not the full poll interval, and no minimum floor was added.
    assert sleeps, "the pre-done poll must have slept once"
    assert sleeps[0] == pytest.approx(remaining)
    assert sleeps[0] < poll
    # After the clock passed the original deadline, NO further sleep starts.
    assert expired_sleeps == [], f"sleep after expiry: {expired_sleeps}"
    assert clock["now"] == pytest.approx(1000.0 + remaining)
    # Every sleep is capped by the remaining budget captured immediately
    # BEFORE that sleep, against the ORIGINAL deadline. The representable
    # float of 1000.0 + remaining differs from the nominal remaining by a
    # few ulps after subtraction, so only a bounded meaningful epsilon is
    # tolerated here; the whole elapsed bound itself is NOT loosened.
    assert all(s <= remaining + 1e-9 for s in sleeps)


def test_snapshot_during_midload_is_coherent_and_never_crashes(tmp_path: Path):
    manifest, sha = _build_corpus(tmp_path)
    # Ground truth for the loader's ``admitted_bytes`` total comes from the
    # ACTUAL fixture-generated regular WAV files on disk, observed by stat/
    # read BEFORE the loader thread starts. The production loader accumulates
    # the FULL WAV byte length per item (``aggregate += len(data)`` where
    # ``data`` is the ENTIRE file, header included), so the expected total is
    # the real on-disk file size -- NOT the PCM payload length and NOT a
    # hardcoded container-header constant. Reading one generated file and
    # stat-ing them all is independent of (non-self-referential to) any later
    # loader snapshot.
    corpus_payload = json.loads(manifest.read_bytes())
    corpus_item_paths = [Path(item["path"]) for item in corpus_payload["items"]]
    corpus_wav_bytes = corpus_item_paths[0].read_bytes()
    expected_wav_file_bytes = len(corpus_wav_bytes)
    # Every frozen item is the SAME regular WAV file on disk.
    assert [p.stat().st_size for p in corpus_item_paths] == [
        expected_wav_file_bytes
    ] * FROZEN_COUNT
    # The real generated file genuinely carries the PCM payload plus a real
    # container, so the expected total is strictly larger than the payload.
    assert corpus_wav_bytes.endswith(PCM_BYTES)
    assert expected_wav_file_bytes > len(PCM_BYTES)
    expected_bytes = expected_wav_file_bytes

    class SlowishWrapper:
        def __init__(self, path: str) -> None:
            self._handle = open(path, "rb")

        def read(self, n: int) -> bytes:
            time.sleep(0.001)  # widen the race window for the reader side
            return self._handle.read(n)

        def close(self) -> None:
            self._handle.close()

        @property
        def closed(self) -> bool:
            # Honest proof delegated to the real underlying handle.
            return self._handle.closed

    opener = FakeOpener(lambda path: SlowishWrapper(path))
    config = _config(tmp_path, manifest, sha)
    loader = gate.OwnedInputLoader(config, opener=opener)
    loader.start()
    counts: list[int] = []
    try:
        while not loader.done_event.is_set():
            # Concurrent reader during ACTUAL writer publication. The writer
            # publishes admitted items AND running totals under the SAME lock
            # this snapshot takes, so every observed view must be coherent.
            snap = loader.snapshot()  # concurrent reader during writer updates
            n = len(snap["admitted"])
            counts.append(n)
            assert len(snap["open_wrappers"]) <= FROZEN_COUNT + 1
            assert len(snap["failures"]) == 0
            # Coherent corpus/caps/totals publication: once caps exist the
            # corpus must exist too, and the running totals must match the
            # admitted count exactly (no torn admitted/totals view).
            if snap["caps"]:
                assert snap["corpus"] is not None
                assert snap["admitted_bytes"] is not None
            if n > 0:
                assert snap["admitted_bytes"] is not None
                # Each item is one WAV of identical size in this corpus.
                # A torn view would pair n items with a totals value that
                # does not match; the shared lock prevents that.
                assert snap["admitted_audio_s"] is not None
            # Recompute the byte total OUTSIDE any lock from the snapshot and
            # confirm it equals the writer-published running total.
            recomputed = sum(len(a.wav_bytes) for a in snap["admitted"].values())
            if snap["admitted_bytes"] is not None:
                assert recomputed == snap["admitted_bytes"]
            time.sleep(0.0005)
    finally:
        # Teardown runs EVEN ON assertion failure: cooperatively cancel and
        # bound the join so this daemon loader cannot keep performing file
        # I/O into later tests.
        loader.cancel_event.set()
        loader.join(timeout=10)
    assert not loader.is_alive()
    assert counts == sorted(counts), "snapshot admitted count must be monotonic"
    assert max(counts) <= FROZEN_COUNT
    assert loader.cleanup_proven() is True
    final = loader.snapshot()
    assert len(final["admitted"]) == FROZEN_COUNT
    # FILE guard: the production total is the full on-disk WAV byte length.
    assert final["admitted_bytes"] == FROZEN_COUNT * expected_bytes
    # PAYLOAD guard (independent of the file-size expectation): the decoded
    # PCM total is still exactly one payload per item. This keeps the guard
    # from being weakened by measuring against the same quantity twice.
    assert sum(len(a.pcm) for a in final["admitted"].values()) == FROZEN_COUNT * len(
        PCM_BYTES
    )
    assert final["admitted_audio_s"] is not None


def test_default_opener_fd_not_leaked_when_wrapper_creation_fails(
    tmp_path: Path, monkeypatch
):
    """Actual injected wrapper-creation failure: when the buffered wrapper
    cannot be created, the SPECIFIC still-owned raw fd is closed exactly once
    on the owning thread, and no second/foreign close is attempted."""
    victim = tmp_path / "one.wav"
    _write_wav(victim, PCM_BYTES)

    real_os = os
    opened: list[int] = []
    closed: list[int] = []

    class FakeOs:
        O_RDONLY = real_os.O_RDONLY
        O_NONBLOCK = getattr(real_os, "O_NONBLOCK", 0)

        @staticmethod
        def open(path, flags, *args, **kwargs):
            fd = real_os.open(path, flags, *args, **kwargs)
            opened.append(fd)
            return fd

        @staticmethod
        def fstat(fd):
            return real_os.fstat(fd)

        @staticmethod
        def close(fd):
            closed.append(fd)
            return real_os.close(fd)

    def failing_open(*args, **kwargs):
        raise OSError("simulated wrapper creation failure")

    monkeypatch.setattr(gate, "os", FakeOs)
    # Gate-LOCAL injection: shadowing ``open`` in the gate module's namespace
    # intercepts exactly the buffered-wrapper creation inside
    # _default_owned_open while every real tool (builtins.open, os.*) stays
    # untouched for the rest of the process.
    monkeypatch.setattr(gate, "open", failing_open, raising=False)

    with pytest.raises(OSError, match="simulated wrapper creation failure"):
        gate._default_owned_open(str(victim))

    assert len(opened) == 1, "exactly one owning-thread open expected"
    owned_fd = opened[0]
    # The SPECIFIC still-owned fd was closed exactly once (no retry, no
    # duplicate/foreign close of some other descriptor).
    assert closed == [owned_fd]
    assert closed.count(owned_fd) == 1
    # OS-level proof, READ-ONLY: fstat on the just-closed SAME descriptor
    # must fail with EBADF. No second raw close (that would be a foreign
    # close attempt) and no /proc/self/fd listing: listdir itself opens a
    # transient descriptor that may reuse the just-closed number, so numeric
    # membership there is not a closure proof.
    with pytest.raises(OSError) as excinfo:
        real_os.fstat(owned_fd)
    assert excinfo.value.errno == errno.EBADF, excinfo.value
