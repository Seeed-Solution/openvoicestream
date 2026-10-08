import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "kokoro_g2p_56_golden_verify.py"
GOLDEN = ROOT / "corpus/kokoro_g2p_56_golden.json"


def test_frozen_golden_shape_and_immutability():
    rows = json.loads(GOLDEN.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 56
    assert all(set(("phonemes", "final_tokens", "preprocessed_text", "tokenizer_tokens", "accepted", "ids")) <= row.keys() for row in rows)


def test_real_fresh_default_g2p_parity(tmp_path):
    python = os.environ.get("KOKORO_G2P_TEST_PYTHON")
    tokens = os.environ.get("KOKORO_G2P_TEST_TOKENS")
    if not python or not tokens:
        pytest.skip("explicit isolated runtime and product tokenizer required")
    report = tmp_path / "report.json"
    proc = subprocess.run([python, "-B", str(SCRIPT), "--tokens", tokens, "--output", str(report)], text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout[-4000:]
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["golden_count"] == data["pass_count"] == 56
    assert data["diffs"] == []
    assert data["tokens_checked"] is True
    assert all(row["checks"]["ids"] for row in data["rows"])


def test_full_verification_requires_tokenizer():
    proc = subprocess.run([sys.executable, str(SCRIPT)], text=True, capture_output=True)
    assert proc.returncode == 2
    assert "--tokens" in proc.stderr


def test_tokens_text_preserves_space_token(tmp_path):
    loader = runpy.run_path(str(SCRIPT))["load_tokens"]
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("a 10\n  16\n", encoding="utf-8")
    assert loader(tokens) == {"a": 10, " ": 16}
