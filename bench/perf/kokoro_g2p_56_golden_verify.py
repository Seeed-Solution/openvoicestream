#!/usr/bin/env python3
"""Run the real Misaki English G2P and compare against the frozen 56-row golden."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import socket
import sys
from pathlib import Path


def _deny(*_args, **_kwargs):
    raise RuntimeError("NETWORK_DISABLED")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _token(t):
    return {"text": t.text, "tag": t.tag, "phonemes": t.phonemes,
            "whitespace": getattr(t, "whitespace", None)}


def _ids(phonemes: str, vocab: dict[str, int]) -> tuple[list[int], list[str]]:
    ids, unknown = [], []
    for char in phonemes:
        key = " " if char.isspace() else char
        if key in vocab:
            ids.append(vocab[key])
        else:
            unknown.append(char)
    return ids, unknown


def load_tokens(path: Path) -> dict[str, int]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(raw)
        vocab = data.get("vocab", data.get("token_to_id", data))
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError("JSON tokenizer map must contain a non-empty vocab")
        return {str(k): int(v) for k, v in vocab.items()}
    vocab = {}
    for line in raw.splitlines():
        token, separator, ident = line.rpartition(" ")
        if not separator or not token:
            raise ValueError("tokens.txt rows must contain a token followed by a space and ID")
        vocab[token] = int(ident)
    if not vocab:
        raise ValueError("empty tokenizer map")
    return vocab


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path,
                        default=here / "corpus/kokoro_g2p_56_golden.json")
    parser.add_argument("--tokens", type=Path, required=True,
                        help="Frozen product tokens.txt or config/tokens/vocab JSON; required for ID parity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    socket.socket.connect = _deny
    socket.socket.connect_ex = _deny
    socket.create_connection = _deny
    socket.getaddrinfo = _deny
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    rows = golden["rows"]
    if len(rows) != 56 or {r.get("route") for r in rows} != {"en-US", "en-GB"}:
        raise ValueError("golden must contain exactly 56 en-US/en-GB rows")
    if len({r["id"] for r in rows}) != 56:
        raise ValueError("golden row IDs are not unique")

    import spacy  # noqa: PLC0415
    from misaki import en  # noqa: PLC0415
    from misaki.espeak import EspeakFallback  # noqa: PLC0415

    if _version("misaki") != "0.9.4":
        raise RuntimeError("Misaki 0.9.4 required")
    if _version("spacy") != "3.8.7":
        raise RuntimeError(f"fresh default spaCy 3.8.7 required, got {_version('spacy')}")
    model_version = _version("en-core-web-sm") or _version("en_core_web_sm")
    if model_version != "3.8.0+kokoropos.1":
        raise RuntimeError(f"slim POS model required, got {model_version}")
    vocab = load_tokens(args.tokens)

    g2ps = {route: en.G2P(trf=False, british=route == "en-GB",
                          fallback=EspeakFallback(route == "en-GB"), unk="")
            for route in ("en-US", "en-GB")}
    result_rows, diffs = [], []
    for expected in rows:
        g2p = g2ps[expected["route"]]
        phonemes, tokens = g2p(expected["text"])
        processed = en.G2P.preprocess(expected["text"])[0]
        doc = g2p.nlp(processed)
        actual = {"phonemes": phonemes,
                  "final_tokens": [_token(t) for t in tokens],
                  "preprocessed_text": processed,
                  "tokenizer_tokens": [{"text": t.text, "tag": t.tag_,
                                         "idx": t.idx, "whitespace": t.whitespace_}
                                        for t in doc]}
        checks = {field: actual[field] == expected[field]
                  for field in actual}
        if vocab is not None:
            actual_ids, unknown = _ids(phonemes, vocab)
            checks["ids"] = not unknown and actual_ids == expected["ids"]
            actual["ids"] = actual_ids
            actual["unknown"] = unknown
        passed = all(checks.values())
        result_rows.append({"id": expected["id"], "route": expected["route"],
                            "checks": checks, "pass": passed, "actual": actual})
        if not passed:
            diffs.append({"id": expected["id"],
                          "failed_fields": [k for k, ok in checks.items() if not ok]})
    report = {"schema": "kokoro.g2p-56-verification.v1", "status": "PASS" if not diffs else "FAIL",
              "golden_sha256": _sha256(args.golden), "golden_count": len(rows),
              "python": sys.executable, "versions": {"misaki": _version("misaki"),
              "spacy": _version("spacy"), "en_core_web_sm": model_version},
              "tokens_checked": vocab is not None, "pass_count": len(rows) - len(diffs),
              "tokens_path": str(args.tokens.resolve()), "tokens_sha256": _sha256(args.tokens),
              "diff_count": len(diffs), "diffs": diffs, "rows": result_rows}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())
