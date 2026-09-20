"""Runtime profile loader for deploy-time backend selection.

Profiles set environment defaults before backend modules are imported.
Operator-supplied env (present at process start) is preserved across reloads;
profile-applied keys are tracked so a subsequent ``apply_profile()`` cleans
stale keys before writing the new profile's defaults.
"""

from __future__ import annotations

import json
import logging
import os
import string
import threading
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

# Families of keys the operator owns, matched by prefix. This table only has to
# cover cross-backend infrastructure: per-backend keys are DERIVED below from
# what the profiles and the leaf registry declare, so adding a backend no longer
# means remembering to edit a list here.
#
# It used to: a missing prefix does not fail loudly, the profile simply
# overwrites the operator's value and nothing is logged. `-e WHISPER_LANGUAGE=zh`
# on the container read back as the profile's `en` with no indication why
# (measured 2026-08-28 on RK3588), and `PIPER_*` set in compose was silently
# taken over by the profile the same way (measured 2026-09-18).
# Kept in full on purpose: derivation only ever WIDENS ownership. Dropping a
# family here would silently un-protect an operator key that no shipped profile
# happens to declare, which is the same failure in the other direction.
_OPERATOR_KEY_PREFIXES: tuple[str, ...] = (
    "OVS_", "LANGUAGE_MODE", "MODEL_DIR", "ASR_", "TTS_", "EDGE_LLM_",
    "QWEN3_", "KOKORO_", "MATCHA_", "RK_", "RKLLM_", "SHERPA_",
    "STREAMING_", "VOCOS_", "HF_", "CUDA_", "TRT_", "NVIDIA_", "WHISPER_",
)


# Process-level variables that are never configuration, however they appear in a
# profile. LD_LIBRARY_PATH is the one that matters: the image sets it
# (Dockerfile.rk) and configs/profiles/jetson-qwen3asr-moss-nx.json deliberately
# replaces it to point the MOSS worker at its own libs. Treating it as
# operator-owned would keep the image value and break that profile.
_PROCESS_VARS: frozenset[str] = frozenset({
    "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME",
    "HOME", "HOSTNAME", "PWD", "SHLVL", "TERM", "TZ", "USER", "SHELL", "LANG",
})


def _declared_backend_keys() -> frozenset[str]:
    """Every env key any shipped profile or leaf declares, by exact name.

    This is the self-maintaining half of operator ownership. A key a profile
    writes is by definition a key an operator may want to override, so the
    profiles themselves are the registry — no prefix has to be predicted.

    Deliberately tolerant: this runs at import, before logging is configured
    and in contexts (tests, tools, a trimmed image) where the config tree may
    be absent or partly unreadable. A failure here must not stop the process;
    it only falls back to prefix matching, which is the pre-2026-09 behaviour.
    """
    keys: set[str] = set()
    # Inlined rather than calling _project_root(): this runs at module import,
    # above that definition.
    root = Path(__file__).resolve().parents[2]

    def _candidates(subdir: str, pattern: str) -> list[Path]:  # noqa: D401
        # ``._name.yaml`` AppleDouble sidecars are binary and DO ship inside the
        # RK speech image (rk-20260913.3 carries one per leaf file — its build
        # context was copied from macOS). Reading one raised UnicodeDecodeError
        # at import and took the whole service down with it, so they are skipped
        # by name as well as guarded by the handlers below.
        try:
            return sorted(
                p for p in (root / "configs" / subdir).glob(pattern)
                if not p.name.startswith("._")
            )
        except OSError:
            # An unreadable or vanished config dir must not take the process
            # down from module import; prefix matching still covers ownership.
            return []

    for path in _candidates("profiles", "*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            # Syntactically valid but structurally wrong JSON (a top-level list,
            # or env: "string") would raise AttributeError/TypeError here, which
            # at module import means the service never starts. Shape-check.
            block = doc.get("env") if isinstance(doc, dict) else None
            if isinstance(block, dict):
                keys.update(str(k) for k in block)
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    # Leaf runtime_env, read without a YAML dependency at import time: the
    # registry loader lives in leaf_composition, importing it here would make
    # profile_loader depend on it (and on PyYAML) on every startup path.
    for path in _candidates("leaves", "*.yaml"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        in_env = False
        env_indent = 0
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip())
            if stripped in ("runtime_env:", "runtime_env: {}"):
                in_env = True
                env_indent = indent
                continue
            if in_env:
                if indent <= env_indent:
                    in_env = False
                elif ":" in stripped:
                    name = stripped.split(":", 1)[0].strip().strip("\"'")
                    if name.replace("_", "").isalnum() and name.upper() == name:
                        keys.add(name)
    return frozenset(keys) - _PROCESS_VARS


_DECLARED_BACKEND_KEYS: frozenset[str] = _declared_backend_keys()


def is_operator_key(name: str, *, declared: frozenset[str] | None = None) -> bool:
    """Whether ``name`` is a key the operator owns (profiles never clobber it).

    ``declared`` adds the keys of a profile that is not part of the shipped
    tree — one selected through ``OVS_PROFILE_JSON`` or bind-mounted into the
    image. Without it, an out-of-tree profile for a backend the tree does not
    know yet would still silently win over the operator's env, which is the
    exact case measured on 2026-09-18 (a mounted rk3588-piper.json against
    ``PIPER_*`` in compose).
    """
    if name in _PROCESS_VARS:
        return False
    if name.startswith(_OPERATOR_KEY_PREFIXES) or name in _DECLARED_BACKEND_KEYS:
        return True
    return bool(declared) and name in declared


def _snapshot_operator_env() -> dict[str, str]:
    """Snapshot every env key matching an operator prefix at import time,
    together with its value.

    These keys are considered owned by the operator (or the surrounding
    container/CI environment) and will NEVER be overwritten or cleared by
    ``apply_profile``.

    值也要一起记：``profile_owned_env`` 允许某个 profile 临时压过 operator 值，
    等切到不拥有该键的 profile 时必须还原成这里记的原值。只把键从
    ``_APPLIED_KEYS`` 里清掉是不够的 —— 清理那一步会跳过 operator key（见
    ``apply_profile`` 的 stale-clear），上一个 profile 的值会一直残留。实测
    （2026-08-08）：jetson-qwen3asr-matcha-nx → jetson-edgellm-v091-matcha 之后
    EDGE_LLM_ASR_PLUGIN_PATH 仍是 /opt/edgellm-bin/...，v091 引擎会配上一代插件，
    正是本该修掉的故障模式的反向。

    Empty values are excluded: docker-compose passes declared-but-unset
    variables (e.g. ``QWEN3_ARTIFACT_MANIFEST:`` with no value) into the
    container as empty strings, not unset, and treating those as
    operator-owned would suppress the profile defaults they were meant to
    inherit from.
    """
    return {
        k: v for k, v in os.environ.items()
        if is_operator_key(k) and v != ""
    }


def _snapshot_env_at_import() -> dict[str, str]:
    """Every non-empty, non-process env value present at import.

    Superset of the operator snapshot, kept because ownership for an
    out-of-tree profile can only be decided once that profile is loaded — by
    then ``apply_profile`` has to be able to read back what the operator
    originally set, and this is the only record of it.
    """
    return {
        k: v for k, v in os.environ.items()
        if v != "" and k not in _PROCESS_VARS
    }


def _snapshot_operator_keys() -> frozenset[str]:
    """键集合视图。保留此名以兼容既有调用方与测试。"""
    return frozenset(_snapshot_operator_env())


# Snapshot taken exactly once at module import. 键集合由取值快照派生，
# 两者不可能漂移。
_OPERATOR_SNAPSHOT: dict[str, str] = _snapshot_operator_env()
_OPERATOR_KEYS: frozenset[str] = frozenset(_OPERATOR_SNAPSHOT)
_ENV_AT_IMPORT: dict[str, str] = _snapshot_env_at_import()

# Operator keys discovered at apply time rather than at import: keys an
# out-of-tree profile declares that the operator had already set. They MUST
# persist once discovered. Keeping them only for the one call that found them
# left the key in ``_APPLIED_KEYS`` but out of the next call's ``eff_operator``,
# so switching to a profile that does not declare it restored the operator's
# value in step 0 and then deleted it in step 1's stale-clear — the operator's
# env silently became unset. Reproduced 2026-09-19 with a mounted
# rk3588-piper.json owning PIPER_MODEL_DIR.
_EXTRA_OPERATOR_KEYS: set[str] = set()

# 当前被 profile_owned_env 压过的 operator key。交还时据此还原。
_OWNED_OVERRIDES: set[str] = set()

# Keys for which an operator/profile mismatch is a hard failure (not a soft
# "operator wins" silent override). These keys steer downstream backend
# selection AND model-download routing; a silent mismatch produces
# misleading runtime failures (e.g. profile picks Qwen3 ASR but
# LANGUAGE_MODE=zh_en in env shadows it, so the Qwen3 artifacts never
# download). Fail loud at startup instead.
CRITICAL_KEYS: frozenset[str] = frozenset({
    "LANGUAGE_MODE", "ASR_BACKEND", "TTS_BACKEND",
})

# Keys written by the most recent ``apply_profile`` call. Used to clear stale
# values when reloading a different profile.
_APPLIED_KEYS: set[str] = set()

# Most recently loaded profile dict (empty when none has been applied).
_CURRENT_PROFILE: dict = {}

# Reentrant lock guarding the three pieces of state above.
_PROFILE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _project_root() -> Path:
    # __file__ = <repo>/server/core/profile_loader.py → parents[2] = <repo>
    return Path(__file__).resolve().parents[2]


def _profile_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return _project_root() / "configs" / "profiles" / candidate.name


def _select_profile_ref() -> str | None:
    """Resolve a profile reference from env (legacy ``apply_profile_from_env`` path)."""
    profile_ref = _env("OVS_PROFILE_JSON") or _env("OVS_PROFILE") or _env("OVS_PROFILE_DEFAULT")
    if not profile_ref:
        language_mode = os.environ.get("LANGUAGE_MODE", "").strip()
        if language_mode == "zh_en":
            profile_ref = "jetson-zh-en"
        elif language_mode == "multilanguage":
            profile_ref = "jetson-multilang-highperf"
    if not profile_ref:
        preset = _env("OVS_PRESET")
        if preset:
            from server.core.profile_selector import resolve_profile_name, UnsupportedPreset
            try:
                profile_ref = resolve_profile_name(preset)
            except UnsupportedPreset as exc:
                logger.error("preset %r not supported on this device: %s", preset, exc)
                raise
            logger.info("preset %r → profile %r", preset, profile_ref)
    return profile_ref


def _derive_tts_model_id(profile: dict) -> str | None:
    """Compute the OVS_TTS_MODEL_ID value implied by ``profile``.

    Returns ``None`` when the profile has no usable hint; callers decide
    whether to write it into env (and whether operator ownership overrides).
    """
    model_id = profile.get("tts_model_id")
    if model_id:
        return str(model_id)

    engines = profile.get("required_engines") or []
    tts_keys = ("kokoro", "matcha", "qwen3", "piper")
    for engine in engines:
        mid = engine.get("model_id", "")
        if "asr" not in str(mid).lower() and any(k in str(mid).lower() for k in tts_keys):
            return str(mid)

    if profile.get("tts_backend"):
        logger.warning(
            "Profile %r has no tts_model_id; speaker tables may mis-scope. "
            "Add tts_model_id to your profile JSON.",
            profile.get("name"),
        )
    return None


def _derive_asr_model_id(profile: dict) -> str | None:
    """Compute the canonical ASR model identity exposed by one profile."""
    model_id = profile.get("asr_model_id")
    if model_id:
        value = str(model_id)
    else:
        value = ""
        for engine in profile.get("required_engines") or []:
            candidate = str(engine.get("model_id", ""))
            if "asr" in candidate.casefold():
                value = candidate
                break
    aliases = {
        "qwen3-asr-0.6b": "qwen3-asr",
        "qwen/qwen3-asr-0.6b": "qwen3-asr",
    }
    result = aliases.get(value.casefold(), value) or None
    if result is None and profile.get("asr_backend"):
        logger.warning(
            "Profile %r has no asr_model_id; versioned model discovery is unavailable. "
            "Add asr_model_id to your profile JSON.",
            profile.get("name"),
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def current_profile() -> dict:
    """Return the most recently loaded profile dict (or ``{}``)."""
    with _PROFILE_LOCK:
        return _CURRENT_PROFILE


def get_max_concurrent_sessions() -> int | None:
    """Return the profile's top-level ``max_concurrent_sessions`` (if set).

    Week 1 production hardening: profiles MAY declare a session limit as
    a top-level integer (not under ``env``); env override
    ``OVS_MAX_CONCURRENT_SESSIONS`` always wins. See
    ``docs/archive/specs/prod-hardening-week1.md`` Deliverable 2.

    Returns ``None`` when the profile does not set the field. Raises
    ``ValueError`` if the value is present but not a positive integer.
    """
    with _PROFILE_LOCK:
        raw = _CURRENT_PROFILE.get("max_concurrent_sessions")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"profile.max_concurrent_sessions must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"profile.max_concurrent_sessions must be > 0, got {value}"
        )
    return value


def get_applied_keys() -> frozenset[str]:
    """Return a snapshot of env keys written by the most recent ``apply_profile``."""
    with _PROFILE_LOCK:
        return frozenset(_APPLIED_KEYS)


def apply_profile(
    profile_ref: str | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
    resolve_engines: bool = False,
    kind: str | None = None,
) -> dict:
    """Load a profile and reconcile process env against it.

    Args:
        profile_ref: Profile name or absolute path. ``None`` falls back to env
            resolution (compatible with ``apply_profile_from_env``).
        overrides: Extra env entries layered on top of the profile (respecting
            operator ownership).
        resolve_engines: Reserved for PR4 — when True, future versions will
            call ``engine_resolver`` to materialize artifact paths. Currently
            a no-op.

    Returns:
        The parsed profile dict (or ``{}`` when no profile could be resolved).
    """
    global _CURRENT_PROFILE

    with _PROFILE_LOCK:
        ref = profile_ref if profile_ref else _select_profile_ref()
        if not ref:
            return {}

        path = _profile_path(ref)
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # Compute the desired env from profile JSON (with $VAR expansion).
        # Two-pass expansion lets a profile reference its own keys, e.g.:
        #   QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm
        #   EDGE_LLM_ASR_ENGINE_DIR=${QWEN3_ARTIFACT_ROOT}/engines/...
        # without depending on the current process env having an identically
        # named (and correct) value already set.
        env_block = profile.get("env") or {}
        new_env: dict[str, str] = {}
        for key, value in env_block.items():
            new_env[key] = _expand_with_profile_env(str(value), env_block)

        derived: dict[str, str] = {
            "OVS_PROFILE_NAME": str(profile.get("name", path.stem)),
        }
        tts_model_id = _derive_tts_model_id(profile)
        if tts_model_id is not None:
            derived["OVS_TTS_MODEL_ID"] = tts_model_id
        asr_model_id = _derive_asr_model_id(profile)
        if asr_model_id is not None:
            derived["OVS_ASR_MODEL_ID"] = asr_model_id

        merged: dict[str, str] = {**new_env, **derived}
        if overrides:
            for k, v in overrides.items():
                merged[k] = str(v)

        # Kind-scoped reload (swap only ASR or only TTS): touch ONLY this
        # modality's own env keys. Shared keys (roots/labels used by both
        # backends) and the OTHER kind's keys are left exactly as the last
        # full / other-kind reload set them. Without this, switching ASR to a
        # bundle would stomp OVS_TTS_MODEL_ID / *_TTS_* / MOSS_* etc., and the
        # live TTS backend (whose model_id is read from env) would drift; a
        # failed kind reload's rollback could likewise clobber a concurrently
        # reloaded other kind.
        def _in_scope(k: str) -> bool:
            return kind is None or _key_kind(k) == kind

        if kind is not None:
            merged = {k: v for k, v in merged.items() if _in_scope(k)}

        new_keys = set(merged.keys())

        # Profile-owned operator keys: a profile may declare operator-prefixed
        # keys it FULLY owns (e.g. a CustomVoice profile owning its EDGE_LLM_TTS_*
        # engine wiring). For those, the profile's value overrides the import-time
        # operator snapshot (image-baked defaults / inherited env) instead of being
        # silently shadowed by it. Default (field absent) = no owned keys =
        # byte-identical behavior for every existing profile. Use this only when a
        # profile carries the COMPLETE set of values for those keys; partial
        # profiles that rely on baked defaults must NOT list them.
        owned = {str(k) for k in (profile.get("profile_owned_env") or [])}

        # Keys THIS profile declares that the operator had already set at
        # import, but that no shipped profile/leaf declares — i.e. a profile
        # loaded from OVS_PROFILE_JSON or bind-mounted for a backend the tree
        # does not carry yet. Without this they are not in _OPERATOR_KEYS, so
        # the loop below would overwrite the operator's value and log nothing.
        # Measured 2026-09-18: a mounted rk3588-piper.json silently beat
        # PIPER_MODEL_DIR from compose.
        declared_here = frozenset(new_env)
        extra_operator = {
            k for k in declared_here
            if k in _ENV_AT_IMPORT
            and k not in _OPERATOR_KEYS
            and k not in _EXTRA_OPERATOR_KEYS
            and is_operator_key(k, declared=declared_here)
        }
        if extra_operator:
            for k in sorted(extra_operator):
                _OPERATOR_SNAPSHOT.setdefault(k, _ENV_AT_IMPORT[k])
            _EXTRA_OPERATOR_KEYS.update(extra_operator)
            logger.info(
                "Profile %r declares operator-set keys the shipped tree does not "
                "know (%s); treating them as operator-owned. Ship a profile or "
                "leaf that declares them to make this permanent.",
                derived["OVS_PROFILE_NAME"], ", ".join(sorted(extra_operator)),
            )
        eff_operator = (_OPERATOR_KEYS | frozenset(_EXTRA_OPERATOR_KEYS)) - owned

        # 0. 交还所有权：上一个 profile 用 profile_owned_env 压过、而这个 profile
        #    不再拥有的 operator key，必须还原成导入时的 operator 取值。不能靠
        #    下面的 stale-clear —— 那一步对 operator key 直接 continue，于是旧值
        #    会一直残留（切回 v091 profile 却仍指着上一代插件）。
        relinquished = {
            k for k in _OWNED_OVERRIDES
            if k not in owned and _in_scope(k)
        }
        for k in relinquished:
            # 不变量：只有 _OPERATOR_SNAPSHOT 里有原值的键才会被加进
            # _OWNED_OVERRIDES（见下面 step 2 的记账），所以这里必然取得到。
            _OWNED_OVERRIDES.discard(k)
            os.environ[k] = _OPERATOR_SNAPSHOT[k]
            logger.info(
                "Profile %s relinquished %s; restored operator value %r",
                derived["OVS_PROFILE_NAME"], k, _OPERATOR_SNAPSHOT[k],
            )

        # 1. Clear stale keys (in previous profile but not in this one),
        #    skipping operator-owned keys AND (for a kind-scoped reload) any key
        #    outside this modality's scope — those belong to the other backend.
        stale = _APPLIED_KEYS - new_keys
        for k in stale:
            if k in eff_operator or not _in_scope(k):
                continue
            os.environ.pop(k, None)

        # 2. Write new values, unconditionally overwriting unless operator-owned.
        for k, v in merged.items():
            if k in eff_operator:
                existing = os.environ.get(k)
                if existing != v:
                    if k in CRITICAL_KEYS:
                        raise RuntimeError(
                            f"Profile {derived['OVS_PROFILE_NAME']} requires "
                            f"{k}={v}, but environment has {k}={existing}. "
                            f"Remove {k} from your environment "
                            f"(compose/.env/shell) or change OVS_PROFILE to "
                            f"one that matches."
                        )
                    logger.warning(
                        "Profile %s wants %s=%s but env overrides to %s",
                        derived["OVS_PROFILE_NAME"], k, v, existing,
                    )
                continue
            os.environ[k] = v
            # 记下「这个 profile 压过了一个 operator key」，供下次交还时还原。
            # 以 _OPERATOR_SNAPSHOT 为准而非 _OPERATOR_KEYS：只跟踪我们**记得住
            # 原值**的键，否则交还时无从还原，反而会误删。
            if k in _OPERATOR_SNAPSHOT and k in owned:
                _OWNED_OVERRIDES.add(k)

        # 3. Update bookkeeping. For a kind-scoped reload only replace this
        #    modality's tracked keys; the other kind's stay recorded so a later
        #    full reload can still reconcile them.
        if kind is None:
            _APPLIED_KEYS.clear()
        else:
            for k in [k for k in _APPLIED_KEYS if _in_scope(k)]:
                _APPLIED_KEYS.discard(k)
        _APPLIED_KEYS.update(k for k in new_keys if k not in eff_operator)
        _CURRENT_PROFILE = profile

        if resolve_engines:
            # Inject required_engines[].env_var -> engine_path into os.environ
            # so a backend built right after (e.g. on hot-reload) reads the new
            # profile's engine paths instead of stale ones. resolve_all() is
            # cheap when engines are already built (cache hit). The injected
            # keys are NOT part of the static profile env dict, so we fold them
            # into _APPLIED_KEYS — otherwise the next apply_profile's stale-clear
            # would not remove them and they'd pollute across reloads/rollbacks
            # (cf. backend_manager rollback env-pollution class of bug).
            try:
                from server.core.engine_resolver import resolve_all
                injected = resolve_all(profile, kind=kind)
                _APPLIED_KEYS.update(
                    k for k in injected
                    if k not in _OPERATOR_KEYS and k not in _EXTRA_OPERATOR_KEYS
                )
            except Exception:
                logger.exception(
                    "apply_profile: engine resolution failed; a freshly built "
                    "backend may see stale/empty engine paths"
                )

        logger.info(
            "Applied profile %s from %s (%d env keys; %d stale cleared)",
            derived["OVS_PROFILE_NAME"],
            path,
            len(_APPLIED_KEYS),
            len(stale),
        )
        return profile


def apply_profile_from_env() -> dict:
    """Compatibility shim: resolve profile from env and apply it.

    Resolution order:
      1. ``OVS_PROFILE_JSON`` — explicit path.
      2. ``OVS_PROFILE`` / ``OVS_PROFILE_DEFAULT`` — profile name.
      3. ``LANGUAGE_MODE`` heuristic (``zh_en`` / ``multilanguage``).
      4. ``OVS_PRESET`` — resolved via ``profile_selector``.
    """
    return apply_profile(None)


# ---------------------------------------------------------------------------
# Artifact pre-flight helpers
# ---------------------------------------------------------------------------

# Env-key suffixes that signal "this value is a filesystem path / artifact".
# Used by ``expected_artifact_paths`` / ``find_missing_artifacts`` to decide
# which env entries are worth existence-checking before a profile reload.
#
# Coverage notes (audit 2026-05-21 against configs/profiles/*.json):
#   - _DIR / _ENGINE / _PATH / _BIN / _MANIFEST / _SETTINGS / _FILTERS: original
#     set covers most jetson/rk profile keys.
#   - _ONNX: KOKORO_SPLIT_*_ONNX, MATCHA_SPLIT_ENCODER_ONNX, PARAFORMER_*_ONNX.
#   - _ROOT: QWEN3_ARTIFACT_ROOT, QWEN3_EDGELLM_JETSON_ROOT.
#   - _BASE: KOKORO_MODEL_BASE, MATCHA_MODEL_BASE.
#   - _VOICES: future Kokoro voices.bin path (not currently in any shipped
#     profile, but cheap to include — startswith("/") filter catches misuse).
#   - _TOKENS: future kokoro/matcha tokens.txt path; current profiles use
#     non-path *_STREAM_MAX_SEGMENT_TOKENS=64 (scalar) — caught by the
#     startswith("/") safety filter below.
#   - _LONG: KOKORO_SPLIT_*_ENGINE_LONG (engine paths used for long context).
#
# Safety net: ``_JSON`` matches both ``EDGE_LLM_ASR_MANIFEST_JSON`` (a path)
# and ``OVS_TTS_SPEAKERS_JSON`` (a JSON blob like '{"0":""}', NOT a path).
# We rely on the ``expanded.startswith("/")`` filter to discard non-path
# values regardless of suffix — never remove that filter without rethinking
# how non-path string env entries flow through this helper.
_PATH_LIKE_SUFFIXES: tuple[str, ...] = (
    "_DIR", "_ENGINE", "_PATH", "_BIN", "_MANIFEST",
    "_SETTINGS", "_FILTERS", "_JSON",
    "_ONNX", "_ROOT", "_BASE", "_VOICES", "_TOKENS", "_LONG",
)


def _expand_with_profile_env(value: str, profile_env: Mapping[str, object]) -> str:
    """Expand ``$VAR`` / ``${VAR}`` using ``profile_env`` first, then ``os.environ``.

    Two-pass expansion handles cases where a profile both defines a key AND
    references it from another key, e.g.::

        QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm        # profile-defined
        EDGE_LLM_ASR_ENGINE_DIR=${QWEN3_ARTIFACT_ROOT}/...   # profile-referenced

    With plain ``os.path.expandvars`` the second key would resolve against
    whatever ``QWEN3_ARTIFACT_ROOT`` happened to be in the *current* process
    env (often empty or wrong), defeating dry-run artifact pre-flight.

    Unknown variables are preserved as-is (``safe_substitute`` behaviour),
    so a malformed ``$`` in a value never raises.
    """
    merged: dict[str, str] = {}
    # os.environ wins as the fallback layer for things the profile doesn't define.
    merged.update({k: v for k, v in os.environ.items()})
    # Profile values override env so a profile-declared key resolves to its
    # profile value, not whatever stale value is in the current process env.
    for k, v in profile_env.items():
        if isinstance(v, str):
            merged[k] = v
    try:
        return string.Template(value).safe_substitute(merged)
    except (ValueError, KeyError):
        return value


# Kind attribution for path-like env keys, so a *kind-scoped* reload (swap
# only TTS, or only ASR) validates just that modality's engines and ignores
# the other half of a bundled profile. Keys matching neither marker set are
# treated as shared and always validated. This is the SINGLE source of truth
# for kind attribution — engine_resolver imports _key_kind from here so the two
# kind-scoping call sites can never drift (a missed engine family, e.g. MOSS,
# would otherwise leak the other modality's engines into a kind-scoped reload).
# Every TTS/ASR engine family MUST have a marker here.
_ASR_KEY_MARKERS = ("ASR", "PARAFORMER", "SENSEVOICE")
_TTS_KEY_MARKERS = (
    "TTS", "MATCHA", "KOKORO", "VOCOS", "SPARK", "MOSS", "SPEAKER_ENCODER",
)


def _key_kind(key: str) -> str | None:
    """Classify a path-like env key as 'asr' / 'tts' / None (shared)."""
    ku = key.upper()
    is_asr = any(m in ku for m in _ASR_KEY_MARKERS)
    is_tts = any(m in ku for m in _TTS_KEY_MARKERS)
    if is_asr and not is_tts:
        return "asr"
    if is_tts and not is_asr:
        return "tts"
    return None


def expected_artifact_paths(profile: dict, kind: str | None = None) -> dict[str, str]:
    """Return ``{env_key: expanded_absolute_path}`` for env entries that
    look like filesystem artifacts.

    When ``kind`` is 'tts' or 'asr', keys that belong exclusively to the
    *other* modality are skipped — a kind-scoped reload only needs its own
    engines present, not the paired backend's (see :func:`_key_kind`).

    Variable expansion uses the profile's own env block layered on top of
    ``os.environ`` (see :func:`_expand_with_profile_env`), so profiles that
    self-reference (e.g. ``${QWEN3_ARTIFACT_ROOT}/engines/...``) resolve
    against the profile's own value, not whatever happens to be in the
    current process env.

    Heuristic: include env keys whose suffix matches
    :data:`_PATH_LIKE_SUFFIXES` AND whose expanded value starts with ``/``
    (absolute path only). Non-absolute paths and non-string values are
    skipped — they're either repo-relative (e.g. ``deploy/artifacts/...``)
    or non-path config (e.g. ``OVS_TTS_SPEAKERS_JSON='{"0":""}'``).
    """
    result: dict[str, str] = {}
    env_block = profile.get("env") or {}
    for k, v in env_block.items():
        if not isinstance(v, str):
            continue
        if not any(k.endswith(s) for s in _PATH_LIKE_SUFFIXES):
            continue
        if kind is not None:
            kk = _key_kind(k)
            if kk is not None and kk != kind:
                continue
        expanded = _expand_with_profile_env(v, env_block)
        if expanded.startswith("/"):
            result[k] = expanded
    return result


def find_missing_artifacts(profile: dict, kind: str | None = None) -> list[dict]:
    """Return missing-path records for the given profile.

    Empty list means all expected absolute paths exist on disk. Record
    shape: ``{"env_var": <key>, "path": <expanded>}``. Pass ``kind`` to scope
    the check to one modality (a TTS-only reload skips ASR engine paths).
    """
    paths = expected_artifact_paths(profile, kind=kind)
    missing: list[dict] = []
    for k, p in paths.items():
        if not os.path.exists(p):
            missing.append({"env_var": k, "path": p})
    return missing
