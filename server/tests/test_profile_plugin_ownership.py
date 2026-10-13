"""老 profile 必须能压过镜像烘焙的 EDGE_LLM_ASR_PLUGIN_PATH。

背景（2026-08-07 实测）：v091 运行时镜像把 EDGE_LLM_ASR_PLUGIN_PATH 写成镜像
ENV，指向 /opt/edgellm-v091 那份 31MB 插件。而 EDGE_LLM_ 是 operator 前缀，
profile_loader 把镜像 ENV 视为 operator 所有、永不覆盖，于是 7 个上一代
profile 声明的 /opt/edgellm-bin 插件（70MB，与它们那代引擎配套）根本没生效。
TensorRT 的 engine plan 按名字+版本查 plugin creator，版本不匹配 → ASR worker
静默退出（stderr 为空），表现为 "ASR worker failed to start: "。

修法是让这些 profile 用现成的 profile_owned_env 机制声明拥有该键。

破坏链路验证：把任一 profile 的 profile_owned_env 去掉后，
test_legacy_profiles_own_asr_plugin_path 会失败。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

PROFILES_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"


@pytest.fixture(autouse=True)
def _isolate_env():
    """每个用例跑在干净的 os.environ 上。

    apply_profile 会往 os.environ 写几十个键，而 profile_loader 的 operator 快照
    是在 import 时按当前 env 取的 —— 上一个用例的残留会被下一个用例的 reload
    当成「运维显式设置」，进而改变 profile 覆盖行为。实测：不隔离时
    OVS_PROFILE_NAME 会被当成 operator key，本文件用例的通过与否取决于执行顺序。
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)

# 上一代（非 v091）且使用 trt_edge_llm ASR 的 profile —— 它们的插件路径必须
# 压过镜像默认值才能配上自己那代的引擎。
LEGACY_TRT_ASR_PROFILES = [
    "jetson-qwen3asr-matcha-nx",
    "jetson-qwen3asr-matcha",
    "jetson-qwen3-composition-nx",
    "jetson-multilang-official",
    "jetson-multilang-highperf",
    "jetson-multilang-highperf-nx",
    "jetson-qwen3asr-moss-nx",
]


def _load(name: str) -> dict:
    return json.loads((PROFILES_DIR / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", LEGACY_TRT_ASR_PROFILES)
def test_legacy_profiles_own_asr_plugin_path(name: str) -> None:
    prof = _load(name)
    env = prof.get("env") or {}
    assert "EDGE_LLM_ASR_PLUGIN_PATH" in env, f"{name} 未声明 ASR 插件路径"
    owned = set(prof.get("profile_owned_env") or [])
    assert "EDGE_LLM_ASR_PLUGIN_PATH" in owned, (
        f"{name} 声明了 EDGE_LLM_ASR_PLUGIN_PATH 却没列入 profile_owned_env；"
        f"EDGE_LLM_ 是 operator 前缀，镜像 ENV 会永久压过它"
    )


def test_owned_keys_must_exist_in_env() -> None:
    """profile_owned_env 只能列自己确实提供了值的键 —— 否则会把该键从
    operator 保护里摘出来却不给替代值，等于凭空清掉一个生效配置。"""
    bad = []
    for path in PROFILES_DIR.glob("*.json"):
        prof = json.loads(path.read_text(encoding="utf-8"))
        env = prof.get("env") or {}
        for key in prof.get("profile_owned_env") or []:
            if key not in env:
                bad.append(f"{path.name}: owned={key} 但 env 里没有该键")
    assert not bad, "\n".join(bad)


def test_owned_key_is_restored_when_relinquished(monkeypatch) -> None:
    """老 profile 压过插件路径后，切回不拥有该键的 v091 profile 必须还原。

    codex review 2026-08-08：只加 profile_owned_env 是不够的 —— 被压过的键会留在
    _APPLIED_KEYS 里，而 stale-clear 对 operator key 直接 continue，于是上一个
    profile 的值一直残留。实测 legacy → v091 之后 EDGE_LLM_ASR_PLUGIN_PATH 仍是
    /opt/edgellm-bin/...，v091 引擎会配上一代插件 —— 正是本要修掉的故障的反向。
    """
    image_default = "/opt/edgellm-v091/libNvInfer_edgellm_plugin.so"
    monkeypatch.setenv("EDGE_LLM_ASR_PLUGIN_PATH", image_default)

    import importlib
    import server.core.profile_loader as pl
    pl = importlib.reload(pl)

    legacy = "jetson-qwen3asr-matcha-nx"
    legacy_want = (_load(legacy)["env"])["EDGE_LLM_ASR_PLUGIN_PATH"]

    pl.apply_profile(legacy)
    assert os.environ["EDGE_LLM_ASR_PLUGIN_PATH"] == legacy_want

    pl.apply_profile("jetson-edgellm-v091-matcha")
    assert os.environ["EDGE_LLM_ASR_PLUGIN_PATH"] == image_default, (
        "切回 v091 后未还原镜像默认值，残留了上一代插件路径"
    )

    # 再切一轮，确认不是一次性生效
    pl.apply_profile(legacy)
    assert os.environ["EDGE_LLM_ASR_PLUGIN_PATH"] == legacy_want
    pl.apply_profile("jetson-edgellm-v091-moss")
    assert os.environ["EDGE_LLM_ASR_PLUGIN_PATH"] == image_default


def test_profiles_with_owned_env_own_language_mode() -> None:
    """Profiles that declare profile_owned_env and define LANGUAGE_MODE in env
    must include LANGUAGE_MODE in profile_owned_env.

    Otherwise, when run in a container image with baked ENV LANGUAGE_MODE=zh_en,
    profile_loader treats LANGUAGE_MODE as operator-owned and raises a CRITICAL_KEYS
    mismatch error at startup.
    """
    bad = []
    for path in sorted(PROFILES_DIR.glob("*.json")):
        prof = json.loads(path.read_text(encoding="utf-8"))
        env = prof.get("env") or {}
        if "LANGUAGE_MODE" in env and "profile_owned_env" in prof:
            owned = set(prof.get("profile_owned_env") or [])
            if "LANGUAGE_MODE" not in owned:
                bad.append(
                    f"{path.name} 声明了 LANGUAGE_MODE={env['LANGUAGE_MODE']} "
                    f"且有 profile_owned_env，但未将 LANGUAGE_MODE 列入 profile_owned_env"
                )
    assert not bad, "\n".join(bad)

