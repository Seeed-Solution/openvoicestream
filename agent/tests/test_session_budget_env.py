"""OVS_AGENT_SESSION_MAX_INPUT_TOKENS: the trim budget follows the deployed
LLM's context, not the shared agent-config.yaml (rk3588 + RK1828 Qwen3-4B
holds 2048 tokens; the 7000 default never trimmed and replies went empty)."""
from __future__ import annotations

import pytest

from ovs_agent.config import load_config


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "agent.yaml"
    p.write_text("session_max_input_tokens: 7000\n", encoding="utf-8")
    return p


def test_unset_keeps_yaml_value(cfg_path, monkeypatch):
    monkeypatch.delenv("OVS_AGENT_SESSION_MAX_INPUT_TOKENS", raising=False)
    assert load_config(cfg_path).session_max_input_tokens == 7000


def test_positive_integer_overrides(cfg_path, monkeypatch):
    monkeypatch.setenv("OVS_AGENT_SESSION_MAX_INPUT_TOKENS", " 1200 ")
    assert load_config(cfg_path).session_max_input_tokens == 1200


def test_none_disables_trimming(cfg_path, monkeypatch):
    monkeypatch.setenv("OVS_AGENT_SESSION_MAX_INPUT_TOKENS", "None")
    assert load_config(cfg_path).session_max_input_tokens is None


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "1.5e3"])
def test_invalid_values_are_ignored(cfg_path, monkeypatch, raw):
    monkeypatch.setenv("OVS_AGENT_SESSION_MAX_INPUT_TOKENS", raw)
    assert load_config(cfg_path).session_max_input_tokens == 7000


def test_empty_value_is_unset(cfg_path, monkeypatch):
    # compose passes ${VAR:-} through as an empty string
    monkeypatch.setenv("OVS_AGENT_SESSION_MAX_INPUT_TOKENS", "")
    assert load_config(cfg_path).session_max_input_tokens == 7000
