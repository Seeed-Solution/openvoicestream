#!/usr/bin/env python3
"""Serial HTTP /asr loopback for Piper DP benchmark WAVs.

This records content and request provenance.  It does not score accuracy.
"""
from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("results"), list):
        rows = []
        for bundle in value["results"]:
            if not isinstance(bundle, dict):
                raise ValueError("results entries must be objects")
            nested = bundle.get("rows")
            if isinstance(nested, list):
                for row in nested:
                    if not isinstance(row, dict):
                        raise ValueError("nested benchmark rows must be objects")
                    merged = dict(row)
                    if merged.get("label") is None and bundle.get("label") is not None:
                        merged["label"] = bundle["label"]
                    rows.append(merged)
            else:
                rows.append(bundle)
    elif isinstance(value, dict) and isinstance(value.get("rows"), list):
        rows = value["rows"]
    else:
        raise ValueError("benchmark JSON must contain a list, results[], or rows[]")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("benchmark rows must be objects")
    return rows


def _wav_refs(row: dict[str, Any], sync: bool) -> list[tuple[int, Path]]:
    key = "sync_wavs" if sync else "stream_wavs"
    single = "sync_wav" if sync else "stream_wav"
    refs = row.get(key)
    if refs is None:
        refs = [row.get(single)] if row.get(single) is not None else []
    if not isinstance(refs, list):
        refs = [refs]
    out: list[tuple[int, Path]] = []
    for index, ref in enumerate(refs):
        raw = ref.get("path") if isinstance(ref, dict) else ref
        if not raw:
            raise ValueError(f"selected WAV reference {index} has no path")
        out.append((index, Path(str(raw))))
    return out


def iter_inputs(paths: Iterable[Path], sync: bool) -> Iterable[dict[str, Any]]:
    for benchmark_path in paths:
        try:
            benchmark_sha = sha256_file(benchmark_path)
            rows = load_rows(benchmark_path)
        except Exception as exc:
            yield {"_error": f"benchmark parse failed: {type(exc).__name__}: {exc}",
                   "benchmark_json": str(benchmark_path), "benchmark_sha256": None}
            continue
        if not rows:
            yield {"_error": "benchmark contains no rows", "benchmark_json": str(benchmark_path),
                   "benchmark_sha256": benchmark_sha}
            continue
        for row_index, row in enumerate(rows):
            try:
                refs = _wav_refs(row, sync)
            except Exception as exc:
                yield {"_error": f"WAV reference failed: {type(exc).__name__}: {exc}",
                       "benchmark_json": str(benchmark_path), "benchmark_sha256": benchmark_sha,
                       "row_index": row_index, "case": row.get("id") or row.get("case"),
                       "label": row.get("label")}
                continue
            if not refs:
                yield {"_error": "row has no selected WAV (expected stream_wavs or stream_wav)",
                       "benchmark_json": str(benchmark_path), "benchmark_sha256": benchmark_sha,
                       "row_index": row_index, "case": row.get("id") or row.get("case"),
                       "label": row.get("label")}
                continue
            for rep, wav_path in refs:
                yield {"benchmark_json": str(benchmark_path), "benchmark_sha256": benchmark_sha,
                       "row_index": row_index, "case": row.get("id") or row.get("case"),
                       "label": row.get("label"), "rep": rep, "wav_path": str(wav_path),
                       "wav": wav_path}


def wav_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        channels, sample_width = wav.getnchannels(), wav.getsampwidth()
        pcm = wav.readframes(frames)
    result: dict[str, Any] = {"wav_sha256": sha256_bytes(data), "wav_bytes": len(data),
                              "sample_rate": rate, "samples": frames,
                              "duration_s": frames / rate if rate else None,
                              "channels": channels, "sample_width": sample_width,
                              "peak_abs": None, "rms": None,
                              "saturated_samples": None}
    if sample_width != 2:
        result["audio_diagnostic_note"] = "PCM16 diagnostics omitted: sample_width is not 2"
        return result
    values = array.array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    if values:
        squares = sum(int(value) * int(value) for value in values)
        result["peak_abs"] = max(abs(int(value)) for value in values)
        result["rms"] = (squares / len(values)) ** 0.5
        result["saturated_samples"] = sum(abs(int(value)) >= 32767 for value in values)
    else:
        result["peak_abs"] = 0
        result["rms"] = 0.0
        result["saturated_samples"] = 0
    result["audio_diagnostic_note"] = "PCM16 integer sample statistics; saturation means abs(int16) >= 32767"
    return result


def _multipart(wav_path: Path, body: bytes) -> tuple[bytes, str]:
    boundary = "codex-asr-" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    return head + body + f"\r\n--{boundary}--\r\n".encode(), boundary


def post_asr(base_url: str, wav_path: Path, language: str, timeout: float) -> dict[str, Any]:
    body, boundary = _multipart(wav_path, wav_path.read_bytes())
    query = urllib.parse.urlencode({"language": language})
    url = base_url.rstrip("/") + "/asr?" + query
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started_at, t0 = utc_now(), time.perf_counter()
    status = None
    response_body = b""
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_body = exc.read()
        error = f"HTTPError: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    parsed: Any = None
    parse_error = None
    if response_body:
        try:
            parsed = json.loads(response_body.decode("utf-8"))
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    result = {
        "url": url, "path": "/asr", "language": language,
        "form_fields": {"file": wav_path.name}, "request_timestamp": started_at,
        "http_status": status, "response_sha256": sha256_bytes(response_body),
        "response_body": response_body.decode("utf-8", errors="replace"),
        "elapsed_s": elapsed, "parsed": parsed, "text": parsed.get("text") if isinstance(parsed, dict) else None,
        "backend": parsed.get("backend") if isinstance(parsed, dict) else None,
    }
    if error:
        result["error"] = error
    if parse_error:
        result["parse_error"] = parse_error
    if status != 200:
        result.setdefault("error", f"HTTP status is not 200: {status}")
    if not isinstance(parsed, dict):
        result.setdefault("error", "response JSON is not an object")
    elif not str(parsed.get("text", "")).strip():
        result.setdefault("error", "response JSON has no non-empty text")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8621")
    parser.add_argument("--language", default="English")
    parser.add_argument("--service-manifest", required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sync", action="store_true", help="Use sync_wavs/sync_wav instead of stream WAVs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.service_manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_sha = sha256_bytes(manifest_bytes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "ok": 0, "failed": 0}
    with output.open("w", encoding="utf-8") as stream:
        for item in iter_inputs([Path(x) for x in args.benchmark_json], args.sync):
            counts["total"] += 1
            common = {"case": item.get("case"), "label": item.get("label"), "rep": item.get("rep"),
                      "benchmark_json": item.get("benchmark_json"), "benchmark_sha256": item.get("benchmark_sha256"),
                      "row_index": item.get("row_index"), "service_manifest": manifest,
                      "service_manifest_sha256": manifest_sha}
            if item.get("_error"):
                common.update({"status": "failed", "error": item["_error"]})
            else:
                wav_path = item["wav"]
                common["wav_path"] = str(wav_path)
                try:
                    common.update(wav_info(wav_path))
                    common["request"] = post_asr(args.base_url, wav_path, args.language, args.timeout)
                    req = common["request"]
                    common["status"] = "ok" if req.get("http_status") == 200 and not req.get("error") and not req.get("parse_error") else "failed"
                    if common["status"] == "failed":
                        common["error"] = req.get("error") or req.get("parse_error") or "invalid ASR response"
                except Exception as exc:
                    common.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            counts[common["status"]] += 1
            stream.write(json.dumps(common, ensure_ascii=False) + "\n")
            stream.flush()
        stream.write(json.dumps({"record_type": "summary", "counts": counts}, ensure_ascii=False) + "\n")
        stream.flush()
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
