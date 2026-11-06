"""Thin async client for the seeed-local-voice HTTP/WS surface.

This adapter adds NO model code: it only translates Wyoming events into calls
against an already-running voice service (``SLV_BASE_URL``).

Two endpoints are used:

* ``WS  /asr/stream``  — streaming ASR. Client sends raw int16 mono PCM frames,
  an empty binary frame signals end-of-stream. Server sends JSON
  ``{"type": "partial"|"final", "text": ..., "is_final": bool}``.
* ``POST /tts/stream`` — streaming TTS. Response body is
  ``uint32 LE sample_rate`` followed by raw int16 mono PCM chunks
  (verified live against RK3588 2026-07-30: header 16000, width 2, channels 1).
"""

from __future__ import annotations

import json
import logging
import struct
from typing import AsyncIterator, Optional

import aiohttp

_LOGGER = logging.getLogger(__name__)

# The voice service is int16 mono; the sample rate is announced in-band by
# /tts/stream and is a query parameter on /asr/stream.
PCM_WIDTH = 2
PCM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 16000


class SlvClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=None, sock_read=timeout)

    @property
    def ws_base(self) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://") :]
        return "ws://" + self.base_url[len("http://") :]

    # ---------------------------------------------------------------- ASR ---
    def asr_stream_url(self, *, language: str = "auto", sample_rate: int) -> str:
        # vad=none: Home Assistant owns endpointing (it sends AudioStop when its
        # own VAD/stage decides the utterance ended). Server-side VAD would
        # inject extra finals mid-utterance, which HA's stt.py cannot consume
        # (its loop breaks on the first Transcript).
        return (
            f"{self.ws_base}/asr/stream"
            f"?language={language}&sample_rate={sample_rate}&vad=none"
        )

    # ---------------------------------------------------------------- TTS ---
    async def tts_stream(
        self,
        session: aiohttp.ClientSession,
        text: str,
        *,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> AsyncIterator[tuple[int, bytes]]:
        """Yield ``(sample_rate, pcm_bytes)`` chunks for ``text``.

        The 4-byte sample-rate header is stripped and reported with every
        chunk so the caller can emit a correct Wyoming ``audio-start``.
        """
        payload: dict = {"text": text}
        if speaker_id is not None:
            payload["speaker_id"] = speaker_id
        if speed is not None:
            payload["speed"] = speed

        async with session.post(
            f"{self.base_url}/tts/stream", json=payload, timeout=self._timeout
        ) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if ctype.startswith("application/json"):
                body = await resp.text()
                raise RuntimeError(f"/tts/stream returned JSON error: {body[:200]}")

            header = b""
            sample_rate = DEFAULT_SAMPLE_RATE
            got_header = False
            pending = b""
            async for chunk in resp.content.iter_chunked(4096):
                if not got_header:
                    header += chunk
                    if len(header) < 4:
                        continue
                    sample_rate = struct.unpack("<I", header[:4])[0]
                    got_header = True
                    chunk = header[4:]
                    if not chunk:
                        continue
                pending += chunk
                # Never split an int16 sample across audio-chunk boundaries.
                usable = len(pending) - (len(pending) % PCM_WIDTH)
                if usable:
                    yield sample_rate, pending[:usable]
                    pending = pending[usable:]
            if pending:
                _LOGGER.warning("dropping %d trailing odd byte(s)", len(pending))

    async def capabilities(self, session: aiohttp.ClientSession) -> dict:
        out: dict = {}
        for kind in ("asr", "tts"):
            try:
                async with session.get(
                    f"{self.base_url}/{kind}/capabilities", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    out[kind] = await resp.json()
            except Exception as err:  # pragma: no cover - best effort
                _LOGGER.warning("capabilities probe for %s failed: %s", kind, err)
                out[kind] = {}
        return out


def parse_asr_message(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
