"""On-demand model downloader.

Checks if required models exist for the current LANGUAGE_MODE.
Downloads missing models from CDN on first start; cached in /opt/models volume.

Models baked into the Docker image (zh_en) are always available.
English-only models (Kokoro TTS + Zipformer ASR) are downloaded on demand
when LANGUAGE_MODE=en, keeping the image small for default users.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional
from typing import Mapping

logger = logging.getLogger(__name__)

CDN_BASE = "https://sensecraft-statics.seeed.cc/solution-app/jetson-voice"

# Model registry: {dir_name: (cdn_filename, description)}
MODELS = {
    "zh_en": {
        "matcha-icefall-zh-en": ("models-matcha.tar.gz", "Matcha TTS (zh+en)"),
        "paraformer-streaming": ("models-paraformer.tar.gz", "Paraformer streaming ASR (zh+en)"),
    },
    "en": {
        "kokoro-multi-lang-v1_0": ("kokoro-multi-lang-v1_0.tar.bz2", "Kokoro TTS v1.0 (English, 53 speakers)"),
        "zipformer-en": ("models-zipformer-en.tar.gz", "Zipformer streaming ASR (English)"),
    },
    "shared": {
        "sensevoice": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            "SenseVoice offline ASR (5 languages)",
        ),
    },
}

# Bundle models tied to a selectable backend: (kind, backend_key). Used to
# suppress over-fetching when a profile explicitly selects a *different*
# backend of the same kind — e.g. a Kokoro profile must not pull Matcha, a
# Qwen3 profile must not pull Paraformer, just because they are bundled in
# MODELS[language_mode]. Models not listed here (sensevoice, zipformer) are
# never profile-gated and keep their legacy language_mode behavior.
_BUNDLE_MODEL_BACKEND = {
    "matcha-icefall-zh-en": ("tts", "jetson.matcha_trt"),
    "kokoro-multi-lang-v1_0": ("tts", "jetson.kokoro_trt"),
    "paraformer-streaming": ("asr", "jetson.paraformer_trt"),
}

# Per-model files the freshness check insists on seeing.
# Without this, model dirs that engine_resolver populated with only
# auxiliary subdirs (engines/, onnx/ skeletons) pass the "non-empty"
# heuristic but still miss load-bearing resources such as tokens.txt.
_REQUIRED_FILES = {
    "matcha-icefall-zh-en": ("model-steps-3.onnx", "tokens.txt", "lexicon.txt"),
    "paraformer-streaming": ("encoder.onnx", "tokens.txt"),
    "zipformer-en": ("encoder.int8.onnx", "tokens.txt"),
    "kokoro-multi-lang-v1_0": ("model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt"),
    "sensevoice": ("model.int8.onnx",),
}


_MATCHA_MANIFEST_PATH = Path(__file__).with_name("matcha_artifacts.json")


def _load_matcha_manifest() -> dict:
    """Load and minimally validate the release-owned Matcha artifact lock."""
    try:
        manifest = json.loads(_MATCHA_MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Matcha artifact lock is unavailable: {_MATCHA_MANIFEST_PATH}: {exc}"
        ) from exc
    if manifest.get("model_id") != "matcha-icefall-zh-en":
        raise RuntimeError("Matcha artifact lock has the wrong model_id")
    return manifest


def _sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while chunk := src.read(bufsize):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_locked_file(path: Path, lock: dict, *, label: str) -> None:
    """Fail closed unless ``path`` matches both published size and SHA256."""
    expected_sha = lock.get("sha256")
    expected_size = lock.get("size")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise RuntimeError(f"{label} has no valid published SHA256 lock")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise RuntimeError(f"{label} has no valid published size lock")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )


def _matcha_model_files_valid(model_path: Path) -> bool:
    """Return True only for the exact release-locked Matcha runtime inputs."""
    files = _load_matcha_manifest()["model_bundle"]["required_files"]
    try:
        for name, lock in files.items():
            _verify_locked_file(model_path / name, lock, label=f"Matcha {name}")
    except RuntimeError as exc:
        logger.warning("Matcha model cache is not release-valid: %s", exc)
        return False
    return True


def _detect_tar_mode(filename: str) -> str:
    """Return tar open mode based on filename extension."""
    if filename.endswith(".tar.bz2"):
        return "bz2"
    return "gz"


def _download_and_extract(
    url: str,
    dest_dir: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    """Download a .tar.gz or .tar.bz2 from URL and extract to dest_dir.

    Uses curl (fast, with progress) if available, falls back to Python stdlib.
    """
    compress = _detect_tar_mode(url)

    # Release-locked artifacts must be downloaded in full before extraction.
    # The historical curl|tar path cannot hash the archive and can leave a
    # partially extracted model directory on transport failure.
    if expected_sha256 is not None or expected_size is not None:
        if expected_sha256 is None or expected_size is None:
            raise RuntimeError("verified archive downloads require SHA256 and size")
        suffix = ".tar.bz2" if compress == "bz2" else ".tar.gz"
        with tempfile.TemporaryDirectory(prefix="matcha_download_") as tmpdir:
            archive = Path(tmpdir) / f"artifact{suffix}"
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--progress-bar", url, "-o", str(archive)],
                    check=True,
                )
            else:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "openvoicestream/1.0"}
                )
                with urllib.request.urlopen(req, timeout=600) as resp, archive.open("wb") as out:
                    shutil.copyfileobj(resp, out, length=1 << 20)
            _verify_locked_file(
                archive,
                {"sha256": expected_sha256, "size": expected_size},
                label="Matcha model bundle",
            )
            with tarfile.open(archive, f"r:{compress}") as tar:
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise RuntimeError(f"unsafe Matcha archive member: {member.name}")
                tar.extractall(path=dest_dir)
        return

    if shutil.which("curl"):
        # curl + tar streaming: no temp file, shows progress
        tar_flag = "j" if compress == "bz2" else "z"
        cmd = f'curl -fSL --progress-bar "{url}" | tar x{tar_flag}f - -C "{dest_dir}"'
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Pure Python fallback
        suffix = ".tar.bz2" if compress == "bz2" else ".tar.gz"
        logger.info("  Fetching %s ...", url)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
            resp = urllib.request.urlopen(req, timeout=600)
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    pct = downloaded * 100 // total
                    mb = downloaded // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                    logger.info("  Progress: %d/%d MB (%d%%)", mb, total_mb, pct)
        try:
            logger.info("  Extracting to %s ...", dest_dir)
            with tarfile.open(tmp_path, f"r:{compress}") as tar:
                tar.extractall(path=dest_dir)
        finally:
            os.unlink(tmp_path)


def ensure_models(
    language_mode: str = "zh_en",
    model_dir: str = "/opt/models",
    qwen3_required_files: list[str] | None = None,
) -> None:
    """Ensure all required models for the given language mode are present.

    Routing is profile-driven first, language_mode-driven second. When a
    profile is loaded, its ``asr_backend`` / ``tts_backend`` fields decide
    which backend-specific artifacts to fetch (Qwen3 ASR, Matcha TTS,
    Kokoro TTS). Profile-triggered requirements are UNIONED with the
    legacy language_mode requirements so callers without a profile keep
    working unchanged (e.g. plain ``LANGUAGE_MODE=en``).
    """
    try:
        from server.core.profile_loader import current_profile
        profile = current_profile() or {}
    except Exception:
        profile = {}

    asr_backend = profile.get("asr_backend")
    tts_backend = profile.get("tts_backend")
    # Opt-in, per-profile ASR artifact manifest (repo-relative path). Its
    # presence means "this profile's trt_edge_llm artifacts are a flat HF file
    # list I declare myself" and routes AWAY from the qwen3 artifact-set
    # machinery, which selects by profile-name family (nx/nano) and can neither
    # match nor serve the v090 layout. Absent → legacy qwen3 path, unchanged.
    asr_artifact_manifest = profile.get("asr_artifact_manifest")
    profile_qwen_files = _profile_qwen_required_files(profile)
    effective_qwen_files = sorted(
        set(qwen3_required_files or []) | set(profile_qwen_files)
    ) or None

    profile_model_sources = _ensure_profile_model_artifacts(profile)

    # Profile-driven extras (UNIONed with language_mode-driven requirements
    # further down). Pure profile users (no LANGUAGE_MODE set) end up with
    # only the entries triggered here.
    extra_required: dict = {}
    matcha = MODELS.get("zh_en", {}).get("matcha-icefall-zh-en")
    kokoro = MODELS.get("en", {}).get("kokoro-multi-lang-v1_0")
    matcha_model_cached = "matcha-icefall-zh-en" in profile_model_sources
    if tts_backend == "jetson.matcha_trt" and matcha and not matcha_model_cached:
        extra_required["matcha-icefall-zh-en"] = matcha
        # Slim image: the SPLIT_TRT acoustic path needs standalone onnx/ files
        # that neither engine_resolver nor the sherpa CDN tarball provide.
        # Pull them from HF here (idempotent + fail-closed; no-op unless the
        # profile selects MATCHA_ACOUSTIC_EP=SPLIT_TRT).
        matcha_base = os.environ.get("MATCHA_MODEL_BASE") or os.path.join(model_dir, "matcha-icefall-zh-en")
        _ensure_matcha_split_onnx(matcha_base)
    if tts_backend == "jetson.kokoro_trt" and kokoro:
        extra_required["kokoro-multi-lang-v1_0"] = kokoro
    qwen_asr_cached = bool(
        {"qwen3-asr", "qwen3-asr-0.6b"} & profile_model_sources
    )
    if asr_backend == "jetson.trt_edge_llm" and not qwen_asr_cached:
        # Mutually exclusive by design: firing both would make a manifest-driven
        # profile ALSO attempt the 26-file qwen3 set (wrong repo, wrong on-disk
        # layout, and for v090 an outright "Cannot pick HF artifact set" abort).
        if asr_artifact_manifest:
            _ensure_edgellm_v090_artifacts(asr_artifact_manifest)
        else:
            # Qwen3 artifacts are deployed via an external script, not via the
            # MODELS/CDN tarball mechanism — fire it as a side-effect here.
            _ensure_qwen3_artifacts(effective_qwen_files)
    if tts_backend == "jetson.moss_tts_nano" and "moss-tts-nano" not in profile_model_sources:
        # MOSS engines + codec + worker are a flat HF file list (not a
        # host-keyed engine bundle), so they bypass the MODELS/CDN tarball
        # mechanism AND engine_resolver. Provision them as a side-effect here,
        # mirroring the Qwen3 dispatch above. engine_resolver still runs after
        # this for the compile-fallback path; its list-shaped-manifest skip
        # (97a9b9f) is untouched.
        _ensure_moss_artifacts()
    if asr_backend == "jetson.sensevoice_trt":
        # SenseVoice on Jetson = standalone TRT engine. Fetch the rescaled fixed
        # ONNX + decode assets from HF and build the engine with the host-mounted
        # TensorRT (so it matches the runtime). Idempotent (skips if built).
        _ensure_sensevoice_trt_artifacts()
    if asr_backend in _WHISPER_ENCODER_FILES:
        # One class, three encoder execution paths; the spec picks which encoder
        # to fetch and WHISPER_VARIANT picks within it. The decoder is shared
        # across every path except Hailo's tiny. Idempotent.
        _ensure_whisper_artifacts(asr_backend)
    if os.environ.get("ASR_BACKEND") == "sensevoice_rknn":
        # SenseVoice RKNN model + decode assets are a flat HF file list; fetch
        # the RK_PLATFORM-specific .rknn + decode assets so switching to a
        # *-sensevoice profile auto-provisions the model. Idempotent.
        _ensure_sensevoice_rknn_artifacts()

    if language_mode == "rk":
        _ensure_rk_artifacts()
        if os.environ.get("RK_ENSURE_MATCHA_RESOURCES", "1").lower() in ("0", "false", "no"):
            return
        required = {"matcha-icefall-zh-en": matcha} if matcha else {}
        required.update(extra_required)
        model_dir = os.environ.get("TTS_MODEL_DIR") or model_dir

    elif language_mode == "multilanguage":
        # Preserve legacy behavior: multilanguage mode triggers Qwen3
        # artifacts even when no profile is loaded. When a profile is
        # active, an explicit Qwen ASR model source has already been
        # provisioned above and must not fall through to the inherited
        # aggregate artifact set. A legacy manifest-driven profile is likewise
        # already provisioned and must remain mutually exclusive with Qwen.
        if not qwen_asr_cached and not asr_artifact_manifest:
            _ensure_qwen3_artifacts(effective_qwen_files)
        required: dict = {}
        # Some multilanguage profiles pair Qwen3 ASR with Matcha TTS. Only
        # those need the Matcha acoustic ONNX + lexicon; pure Qwen3 profiles
        # should not download or validate Matcha assets during startup.
        if tts_backend == "jetson.matcha_trt" and matcha and not matcha_model_cached:
            required["matcha-icefall-zh-en"] = matcha
        required.update(extra_required)
        if not required:
            return
    else:
        required = {}
        required.update(MODELS.get(language_mode, {}))
        if os.environ.get("ENSURE_OFFLINE_ASR", "").lower() in ("1", "true", "yes"):
            required.update(MODELS.get("shared", {}))
        # Profile-driven suppression of the language_mode bundle: when a profile
        # explicitly selects backends, a bundled model tied to a *different*
        # backend of the same kind (ASR/TTS) is not needed and must not be
        # fetched. Restores the per-backend exclusivity that 9cc1f35 lost when it
        # switched to UNION routing (which over-fetched Matcha for a Kokoro
        # profile, or Paraformer for a Qwen3 profile). Backward-compatible: pure
        # LANGUAGE_MODE deployments (no profile backends) skip the filter.
        if asr_backend or tts_backend:
            for dir_name in list(required):
                kind_backend = _BUNDLE_MODEL_BACKEND.get(dir_name)
                if kind_backend is None:
                    continue  # not a profile-gated backend model
                kind, backend = kind_backend
                selected = asr_backend if kind == "asr" else tts_backend
                if selected != backend:
                    required.pop(dir_name, None)
        required.update(extra_required)
    if not required:
        return

    missing = []
    for dir_name, (cdn_file, desc) in required.items():
        model_path = os.path.join(model_dir, dir_name)
        required_files = _REQUIRED_FILES.get(dir_name)
        # When required files are declared, look for the actual load-bearing
        # files recursively under the model dir (the tarball lays files
        # under subdirs in some upstream variants). Non-empty dir alone
        # is NOT a sufficient signal — engine_resolver may have written
        # the engines/ subdir before model_downloader runs.
        is_ready = False
        if os.path.isdir(model_path):
            if dir_name == "matcha-icefall-zh-en":
                is_ready = _matcha_model_files_valid(Path(model_path))
            elif required_files:
                found = set()
                # followlinks: a model dir is often a symlink to the extracted
                # tarball (e.g. /opt/models/sensevoice/sherpa-onnx-sense-voice-*
                # -> ../sherpa-onnx-sense-voice-*). os.walk does not descend
                # into symlinked dirs by default, so the present model reads as
                # missing, and startup then dies trying to re-download it.
                #
                # Following links reintroduces the cycle os.walk avoids by
                # default: one link back to an ancestor (or two dirs linking to
                # each other) and the walk never terminates — a hung startup,
                # strictly worse than the missing-model bug above. Prune on
                # realpath rather than bounding depth, which a short cycle would
                # still spin inside of.
                seen_dirs: set[str] = set()
                for root, dirs, files in os.walk(model_path, followlinks=True):
                    real_root = os.path.realpath(root)
                    if real_root in seen_dirs:
                        dirs[:] = []  # already walked via another path
                        continue
                    seen_dirs.add(real_root)
                    found.update(name for name in required_files if name in files)
                is_ready = found == set(required_files)
            elif os.listdir(model_path):
                is_ready = True
        if is_ready:
            logger.info("Model OK: %s (%s)", dir_name, desc)
        else:
            missing.append((dir_name, cdn_file, desc))

    if not missing:
        logger.info("All models for mode '%s' are ready.", language_mode)
        if language_mode == "en" or "kokoro-multi-lang-v1_0" in required:
            _patch_kokoro_voices(model_dir)
        return

    logger.info(
        "Downloading %d missing model(s) for mode '%s'...",
        len(missing), language_mode,
    )

    os.makedirs(model_dir, exist_ok=True)

    for dir_name, cdn_file, desc in missing:
        # SenseVoice (RPi/CPU sherpa tarball) defaults to a raw GitHub release
        # with no CDN fallback — impractically slow on edge devices without good
        # GitHub access. Default to the HF copy (honors HF_ENDPOINT, so RPi via
        # hf-mirror is fast); SENSEVOICE_MODEL_URL overrides, SENSEVOICE_RKNN_HF_REPO
        # picks the repo.
        if dir_name == "sensevoice":
            url = os.environ.get("SENSEVOICE_MODEL_URL")
            if not url:
                endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
                repo = os.environ.get("SENSEVOICE_RKNN_HF_REPO", "harvestsu/sensevoice-rknn")
                url = f"{endpoint}/{repo}/resolve/main/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
        # Use GitHub releases for models not hosted on CDN
        elif cdn_file.startswith("http"):
            url = cdn_file
        elif cdn_file == "kokoro-multi-lang-v1_0.tar.bz2":
            url = os.environ.get(
                "KOKORO_MODEL_URL",
                f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{cdn_file}",
            )
        else:
            url = f"{CDN_BASE}/{cdn_file}"
        logger.info("Downloading %s ...", desc)
        try:
            if dir_name == "matcha-icefall-zh-en":
                bundle = _load_matcha_manifest()["model_bundle"]
                if url != bundle["url"]:
                    raise RuntimeError(
                        "Matcha model URL differs from the release-owned integrity lock"
                    )
                _download_and_extract(
                    url,
                    model_dir,
                    expected_sha256=bundle["sha256"],
                    expected_size=bundle["size"],
                )
                if not _matcha_model_files_valid(
                    Path(model_dir) / "matcha-icefall-zh-en"
                ):
                    raise RuntimeError(
                        "Matcha archive extracted but required files failed integrity"
                    )
            else:
                _download_and_extract(url, model_dir)
            logger.info("Downloaded %s OK.", desc)
        except Exception as e:
            logger.error("Failed to download %s: %s", desc, e)
            logger.error(
                "You can manually download from %s and extract to %s",
                url, model_dir,
            )
            sys.exit(1)

    if language_mode == "en" or "kokoro-multi-lang-v1_0" in required:
        _patch_kokoro_voices(model_dir)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_profile_model_artifacts(profile: dict) -> set[str]:
    """Provision explicit flat-profile model sources, then legacy composition sources."""
    if not isinstance(profile, dict):
        return set()
    requests: list[dict[str, object]] = []
    declared = profile.get("model_artifacts")
    if declared is not None:
        if not isinstance(declared, (list, tuple)):
            raise RuntimeError("profile.model_artifacts must be a list")
        for raw in declared:
            if not isinstance(raw, Mapping):
                raise RuntimeError("profile.model_artifacts entries must be mappings")
            request = {
                "model_id": str(raw.get("model_id") or raw.get("model") or ""),
                "repo": str(raw.get("repo") or raw.get("hf_repo") or ""),
                "revision": str(raw.get("revision") or "main"),
                "canonical_model_id": str(raw.get("canonical_model_id") or raw.get("canonical_id") or ""),
                "root": str(raw.get("root") or raw.get("model_root") or ""),
                "manifest": str(raw.get("manifest") or raw.get("manifest_path") or ""),
                "cache_root": str(raw.get("cache_root") or raw.get("model_cache_root") or ""),
                "files": [str(path) for path in (raw.get("files") or raw.get("required_files") or ())],
                "strict": bool(raw.get("strict", False)),
            }
            if not request["model_id"] or not request["repo"]:
                raise RuntimeError("profile.model_artifacts entries require model_id and repo")
            requests.append(request)
    if profile.get("composition"):
        try:
            from server.core.composition_boot import model_sources
            sources = model_sources(profile)
        except Exception as exc:
            logger.warning("composition model source resolution unavailable: %s", exc)
            sources = []
        for source in sources:
            if not source.repo or not (source.canonical_model_id or source.root or source.manifest or source.cache_root):
                continue
            requests.append({
                "model_id": source.model_id, "repo": source.repo, "revision": source.revision,
                "canonical_model_id": source.canonical_id, "root": source.root,
                "manifest": source.manifest, "cache_root": source.cache_root,
                "files": list(source.files),
                "strict": source.strict,
            })
    if not requests:
        return set()
    from server.core.qwen3_artifact_downloader import ensure_model_requests
    ensure_model_requests(requests)
    return {str(request.get("canonical_model_id") or request["model_id"]) for request in requests}


def _profile_qwen_required_files(profile: dict) -> list[str]:
    """Return root-relative Qwen engine/model inputs for one active profile."""

    root = Path(
        os.environ.get("QWEN3_ARTIFACT_ROOT", "/opt/models/edgellm-v091")
    )
    paths: list[Path] = []
    for item in profile.get("required_engines", []):
        raw = item.get("engine_path") if isinstance(item, dict) else None
        if raw:
            paths.append(Path(str(raw)))

    env = profile.get("env") or {}
    if profile.get("asr_backend") == "jetson.trt_edge_llm":
        audio_dir = env.get("EDGE_LLM_ASR_AUDIO_ENC_DIR")
        if audio_dir:
            paths.append(Path(str(audio_dir)) / "audio" / "audio_encoder.engine")
    fixed_embedding = env.get("EDGE_LLM_TTS_BASE_SPK_EMBED_PATH")
    if fixed_embedding:
        paths.append(Path(str(fixed_embedding)))

    relative: set[str] = set()
    for path in paths:
        try:
            relative.add(path.relative_to(root).as_posix())
        except ValueError:
            # Matcha and other non-Qwen assets share /opt/models but have their
            # own downloader; never fold them into the Edge-LLM snapshot.
            continue
    return sorted(relative)


def _ensure_qwen3_artifacts(required_files_override: list[str] | None = None) -> None:
    """Verify or download Qwen3 artifacts for the active multilanguage profile.

    The deploy script + manifest live in the sibling `qwen3-edgellm-jetson`
    repo so they are not duplicated here. Set `QWEN3_EDGELLM_JETSON_ROOT` to
    override the default `~/project/qwen3-edgellm-jetson` lookup path.
    """
    if os.environ.get("QWEN3_ARTIFACT_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
        logger.info("Qwen3 artifact auto-download disabled.")
        return

    qej_root = Path(
        os.environ.get(
            "QWEN3_EDGELLM_JETSON_ROOT",
            os.path.expanduser("~/project/qwen3-edgellm-jetson"),
        )
    )
    script = qej_root / "scripts" / "deploy_qwen3_artifacts.py"
    manifest = os.environ.get(
        "QWEN3_ARTIFACT_MANIFEST",
        str(qej_root / "deploy" / "artifacts" / "qwen3_manifest.json"),
    )
    artifact_set = os.environ.get("QWEN3_ARTIFACT_SET") or "orin-nano-highperf-2026-05-10"
    root = os.environ.get("QWEN3_ARTIFACT_ROOT")
    if not script.exists():
        # Slim image: the qwen3-edgellm-jetson submodule COPY was narrowed to
        # deploy/ only, so the deploy script is absent. Fall back to the
        # self-contained in-app HF downloader (qwen3_artifact_downloader) which
        # reads the same manifest (shipped in the slim image at
        # /opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json),
        # picks the matching set, and snapshot_downloads the required engine
        # files from HF. Without this, the slim image silently skipped all
        # qwen3 ASR provisioning and the backend later raised FileNotFoundError.
        logger.warning(
            "Qwen3 artifact deploy script missing at %s — falling back to "
            "in-app HF downloader (slim image path).",
            script,
        )
        _ensure_qwen3_artifacts_via_hf(manifest, artifact_set, required_files_override)
        return

    cmd = [sys.executable, str(script), "--manifest", manifest, "--set", artifact_set]
    if root:
        cmd.extend(["--root", root])
    if os.environ.get("QWEN3_ARTIFACT_VERIFY_SHA256", "1").lower() not in ("0", "false", "no"):
        cmd.append("--verify-sha256")
    logger.info("Ensuring Qwen3 artifact set %s via %s", artifact_set, manifest)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Qwen3 artifact check/download failed with exit code %s", exc.returncode)
        sys.exit(exc.returncode)


# Standalone split-encoder ONNX files for the Matcha SPLIT_TRT acoustic path.
# These live on the HF artifact repo under models/matcha-icefall-zh-en/onnx/.
# The matcha provisioning flow otherwise only (a) extracts a host-keyed engine
# bundle (engine_resolver) and (b) pulls the sherpa CDN tarball — neither of
# which contains these standalone onnx/ files. matcha_trt's SPLIT_TRT path
# hard-requires both at preload (FileNotFoundError otherwise).
_MATCHA_SPLIT_ONNX_FILES = tuple(
    _load_matcha_manifest()["split_onnx"].keys()
)


def _ensure_matcha_split_onnx(model_base: str) -> None:
    """Provision the Matcha split-encoder standalone ONNX files from HF.

    Only relevant for the SPLIT_TRT acoustic path (``MATCHA_ACOUSTIC_EP`` =
    ``SPLIT_TRT``/``TRT_SPLIT``/``HYBRID_TRT``). Idempotent only when present
    files match the release lock. Missing, truncated, or hash-drifted files are
    re-fetched; any download/integrity error aborts startup before preload.
    """
    ep = (os.environ.get("MATCHA_ACOUSTIC_EP") or "").upper()
    if ep not in ("SPLIT_TRT", "TRT_SPLIT", "HYBRID_TRT"):
        return

    # The encoder ONNX env points at .../onnx/matcha_encoder_trt.onnx; derive
    # the onnx dir from it when set, else fall back to <model_base>/onnx.
    enc_env = os.environ.get("MATCHA_SPLIT_ENCODER_ONNX")
    onnx_dir = Path(enc_env).parent if enc_env else Path(model_base) / "onnx"

    locks = _load_matcha_manifest()["split_onnx"]
    targets = {name: onnx_dir / name for name in _MATCHA_SPLIT_ONNX_FILES}
    missing = {}
    for name, dest in targets.items():
        try:
            _verify_locked_file(dest, locks[name], label=f"Matcha split ONNX {name}")
        except RuntimeError:
            missing[name] = dest
    if not missing:
        logger.info("Matcha split-encoder ONNX already present under %s.", onnx_dir)
        return

    logger.info(
        "Matcha split-encoder ONNX provisioning: %d/%d missing under %s — fetching from HF.",
        len(missing), len(targets), onnx_dir,
    )
    try:
        from server.core.hf_artifacts import download_file, ArtifactError
    except Exception as exc:
        raise RuntimeError(
            f"Matcha split ONNX: hf_artifacts unavailable: {exc}"
        ) from exc

    for name, dest in missing.items():
        lock = locks[name]
        rel = lock["repo_path"]
        try:
            download_file(rel, dest, expected_sha256=lock["sha256"])
            _verify_locked_file(dest, lock, label=f"Matcha split ONNX {name}")
            logger.info("Matcha split ONNX downloaded: %s", dest)
        except ArtifactError as exc:
            raise RuntimeError(
                f"Matcha split ONNX download failed for {name}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — integrity failures must abort boot
            raise RuntimeError(
                f"Matcha split ONNX integrity failed for {name}: {exc}"
            ) from exc


def _ensure_qwen3_artifacts_via_hf(
    manifest_path: str,
    artifact_set: str,
    required_files_override: list[str] | None = None,
) -> None:
    """Slim-image fallback: provision Qwen3 ASR artifacts via the in-app HF downloader.

    The fat-image path shells out to ``deploy_qwen3_artifacts.py``; that script
    is absent in the slim image. Here we read the same manifest, determine the
    set's required files and their on-disk dest under ``root``, compute which are
    missing, and call ``qwen3_artifact_downloader.ensure_artifacts`` (which does
    the actual ``snapshot_download`` from HF with allow_patterns derived from the
    required_files).

    Gated by the existing env flags. Fail-open: a download error is logged and
    the backend's own preload-time FileNotFoundError still gates correctness.
    """
    if os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1") != "1":
        logger.info("OVS_AUTO_DOWNLOAD_ARTIFACTS=0 → skipping Qwen3 HF auto-download.")
        return

    # The profile may set QWEN3_ARTIFACT_MANIFEST to a path relative to the
    # qwen3-edgellm-jetson root (e.g. "deploy/artifacts/qwen3_manifest.json"),
    # which does not resolve against the container cwd. Resolve candidates in
    # order: the given path, <QWEN3_EDGELLM_JETSON_ROOT>/<path>, and the known
    # slim-image absolute location.
    qej_root = os.environ.get("QWEN3_EDGELLM_JETSON_ROOT", "/opt/qwen3-edgellm-jetson")
    candidates = [
        Path(manifest_path),
        Path(qej_root) / manifest_path,
        Path("/opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json"),
    ]
    mp = next((c for c in candidates if c.exists()), None)
    if mp is None:
        logger.warning(
            "Qwen3 manifest not found (tried %s) — cannot HF auto-download.",
            [str(c) for c in candidates],
        )
        return
    try:
        manifest = json.loads(mp.read_text())
    except Exception as exc:
        logger.warning("Failed to parse Qwen3 manifest %s (%s).", mp, exc)
        return

    sets = manifest.get("artifact_sets", {})
    set_spec = sets.get(artifact_set)
    if set_spec is None:
        logger.warning(
            "Qwen3 artifact set %r not in manifest %s — skipping HF download.",
            artifact_set, mp,
        )
        return

    root = Path(
        os.environ.get("QWEN3_ARTIFACT_ROOT")
        or set_spec.get("root")
        or "/opt/models/qwen3-edgellm"
    )
    required_files = required_files_override if required_files_override else (set_spec.get("required_files") or [])
    if not required_files:
        logger.warning("Qwen3 set %r declares no required_files — nothing to fetch.", artifact_set)
        return

    # required_files are paths relative to the set root. The profile env
    # (QWEN3_ARTIFACT_ROOT, EDGE_LLM_ASR_ENGINE_DIR, EDGE_LLM_ASR_AUDIO_ENC_DIR)
    # is layered on top of the same root, so root-relative resolution matches.
    expected_paths = [str(root / rf) for rf in required_files]
    missing_paths = [p for p in expected_paths if not Path(p).exists()]
    if not missing_paths:
        logger.info("Qwen3 ASR artifacts already present under %s (%d files).", root, len(expected_paths))
        return

    logger.info(
        "Qwen3 ASR slim provisioning: %d/%d files missing under %s — fetching from HF.",
        len(missing_paths), len(expected_paths), root,
    )
    try:
        from server.core.qwen3_artifact_downloader import ensure_artifacts
        ensure_artifacts(missing_paths)
    except Exception as exc:
        logger.error("Qwen3 ASR HF auto-download failed (%s) — backend preload will re-check.", exc)
        return

    still_missing = [p for p in expected_paths if not Path(p).exists()]
    if still_missing:
        logger.warning(
            "Qwen3 ASR HF download finished but %d files still missing (e.g. %s).",
            len(still_missing), still_missing[0],
        )
    else:
        logger.info("Qwen3 ASR artifacts ready under %s.", root)


def _ensure_rk_artifacts() -> None:
    """Verify or download RK model artifacts when an RK manifest is configured."""
    try:
        from server.core.rk_artifacts import ensure_rk_artifacts
        ensure_rk_artifacts()
    except Exception as exc:
        logger.error("RK artifact check/download failed: %s", exc)
        sys.exit(1)


def _ensure_moss_artifacts() -> None:
    """Verify or download MOSS-TTS-Nano artifacts (slim image runtime provision).

    No-op on the fat image (artifacts baked) when MOSS_ARTIFACT_AUTO_DOWNLOAD
    is disabled. On the slim image, pulls the MOSS engines + codec + worker
    from HF per ``deploy/artifacts/moss_manifest.json``. Idempotent.
    """
    try:
        from server.core.moss_artifacts import ensure_moss_artifacts
        ensure_moss_artifacts()
    except Exception as exc:
        logger.error("MOSS artifact check/download failed: %s", exc)
        sys.exit(1)


def _ensure_edgellm_v090_artifacts(manifest_path: str) -> None:
    """Verify or download the edgellm v0.9.0 ASR artifacts (slim image path).

    ``manifest_path`` is the profile's ``asr_artifact_manifest`` (repo-relative,
    e.g. ``deploy/artifacts/edgellm_v090_manifest.json``). Only the files that
    manifest lists are fetched, which is how the MOSS-TTS profile avoids pulling
    the v090 TTS engines it never loads. Fatal on failure: unlike the optional
    MOSS worker, the ASR engines have no fallback backend.
    """
    try:
        from server.core.edgellm_v090_artifacts import ensure_edgellm_v090_artifacts
        ensure_edgellm_v090_artifacts(manifest_path)
    except Exception as exc:
        logger.error("edgellm v090 ASR artifact check/download failed: %s", exc)
        sys.exit(1)


# SenseVoice RKNN: encoder .rknn (per SoC) + decode assets, hosted as a flat HF
# file list so a *-sensevoice profile auto-provisions on first start.
_SENSEVOICE_RKNN_SHARED = ("am.mvn", "embedding.npy", "chn_jpn_yue_eng_ko_spectok.bpe.model")
# Both SoCs run fp16 with the math-exact K=8 activation rescale on the last
# encoder block's FFN (w_2 weight+bias divided by K, a Div(K) inserted on the
# residual path; LayerNorm is scale-invariant, so accuracy is unchanged and only
# the intermediate range moves back inside fp16). int8 is not an option here —
# it collapses the 25055-way CTC projection.
#
# RK3576 shipped plain fp16 first because it passed a fixed corpus. It does not
# actually hold: the same overflow exists (the block's residual add measures
# 73670 against an fp16 max of 65504) and whether a given clip trips it depends
# on total audio duration, which a fixed corpus never varies. Measured
# 2026-08-25 on RK3576 across 45 duration points: plain fp16 failed 15 and
# mangled one clip at its natural length (`啸计中心也未啸迹象` for
# `而且太平洋海啸预警中心也表示并未发现海啸迹象`, encoder_out max 75.2 vs a
# normal ~37); the rescaled build passed all 45 with magnitudes matching the
# fp32 CPU reference. Keep both platforms on -scaled; do not "simplify" this
# back to a per-SoC split.
#
# The `t172` in the filename is the encoder sequence length the .rknn was frozen
# to: 172 LFR frames = 10.1 s of audio, against the 344 (20.4 s) the first builds
# used. A VAD-delimited utterance is 4-6 s, so the 344 encoder spent more than
# half its NPU time on zero padding. Measured on idle boards, 100-segment corpus:
# single-pass encoder latency RK3588 1120 -> 441 ms and RK3576 675 -> 343 ms, zh
# CER 0.0513 -> 0.0474, and RK3588 p95 over /asr/stream drops under 1.5 s at
# every concurrency from 1 to 8 where the 344 build cleared none. Audio longer
# than one pass is windowed by the backend, not truncated.
_SENSEVOICE_RKNN_FILE = {
    "rk3576": "sense-voice-encoder.rk3576.fp16-scaled.t172.rknn",
    "rk3588": "sense-voice-encoder.rk3588.fp16-scaled.t172.rknn",
}


def _ensure_sensevoice_rknn_artifacts() -> None:
    """Download the SenseVoice RKNN model + decode assets if missing (idempotent).

    Fetches the ``RK_PLATFORM``-specific encoder ``.rknn`` plus the shared decode
    assets (CMVN, prompt embeddings, sentencepiece model) from HF into
    ``SENSEVOICE_RKNN_MODEL_DIR``. Honors HF_ENDPOINT mirrors. The HF repo is
    overridable via ``SENSEVOICE_RKNN_HF_REPO``.
    """
    dest = os.environ.get("SENSEVOICE_RKNN_MODEL_DIR", "/opt/asr/sensevoice-rknn")
    platform = os.environ.get("RK_PLATFORM", "rk3576").lower()
    repo = os.environ.get("SENSEVOICE_RKNN_HF_REPO", "harvestsu/sensevoice-rknn")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{repo}/resolve/main"

    rknn_file = _SENSEVOICE_RKNN_FILE.get(platform, f"sense-voice-encoder.{platform}.fp16.rknn")
    files = [rknn_file, *_SENSEVOICE_RKNN_SHARED]
    os.makedirs(dest, exist_ok=True)
    for name in files:
        path = os.path.join(dest, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info("SenseVoice RKNN asset OK: %s", name)
            continue
        url = f"{base}/{name}"
        logger.info("Downloading SenseVoice RKNN asset %s ...", url)
        tmp = path + ".part"
        try:
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
                     "--retry", "3", "-o", tmp, url],
                    check=True, timeout=1900,
                )
            else:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
                with urllib.request.urlopen(req, timeout=1800) as resp, open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            os.replace(tmp, path)
            logger.info("SenseVoice RKNN asset ready: %s (%d bytes)", name, os.path.getsize(path))
        except Exception as exc:
            logger.error("Failed to download SenseVoice RKNN asset %s: %s", name, exc)
            logger.error("Manually place %s under %s", name, dest)
            raise


# ---------------------------------------------------------------------------
# Whisper (voxedge.backends.whisper): accelerator encoder + CPU KV-cache decoder
# ---------------------------------------------------------------------------
#
# One HF repo carries every platform's encoder plus the two decoder families,
# because the decoder is shared: Hailo's base HEF, both RK .rknn encoders and
# the Jetson ONNX all feed the SAME base decoder pair. Only Hailo's tiny HEF
# needs the tiny pair (4 layers / d384 against base's 6 / d512 — they are not
# interchangeable, and pairing them across families produces fluent nonsense
# rather than an error).
#
# The .rknn and .hef files are compiled per accelerator and ship prebuilt. The
# Jetson side ships ONNX instead: a TRT .plan is version-specific, so it is
# built on-device — and it must be built with `--bf16`, NOT `--fp16`. The fp16
# build of this graph scores cosine 0.826 against onnxruntime and fails
# silently, emitting fluent text that drifts off-topic. See
# docs/perf/whisper-cross-device-20260827.md.
_WHISPER_SHARED = ("mel_80_filters.txt", "vocab_en.txt", "vocab_zh.txt")
_WHISPER_DECODER_BASE = (
    "decoder/base/decoder_model.onnx",
    "decoder/base/decoder_with_past_model.onnx",
)
_WHISPER_DECODER_TINY = (
    "decoder/tiny/decoder_model.onnx",
    "decoder/tiny/decoder_with_past_model.onnx",
)
# Keyed by (spec, variant): the window is a property of the ARTIFACT, not of the
# accelerator. Hailo ships tiny at 10 s and base at 5 s, RK ships 10 s and 20 s
# — so a default keyed only by encoder kind hands the 5 s HEF a 10 s window, and
# rknn-lite does not validate that.
_WHISPER_GEOMETRY = {
    ("hailo.whisper", "tiny"):      (10.0, 1.0),
    ("hailo.whisper", "base"):      (5.0, 1.0),
    ("rk.whisper", "base10"):       (10.0, 0.0),
    ("rk.whisper", "base20"):       (20.0, 0.0),
    ("jetson.whisper_trt", "base"): (30.0, 0.0),
}
# The Jetson artifact that ships is ONNX; what loads is the plan built from it
# on-device. Both sides read this, so the name cannot drift between them.
_WHISPER_TRT_PLAN = "encoder/jetson/enc_base_30s_bf16.plan"

# Keyed by the spec the profile declares, since one class serves three paths.
_WHISPER_ENCODER_FILES = {
    "hailo.whisper": {
        # Hailo publishes tiny at a 10 s window and base at 5 s; both carry a
        # 1 s boundary-hallucination guard, so the usable window is one second
        # shorter than the compiled one.
        "tiny": ("encoder/hailo/tiny-whisper-encoder-10s_15dB.hef", _WHISPER_DECODER_TINY),
        "base": ("encoder/hailo/base-whisper-encoder-5s.hef", _WHISPER_DECODER_BASE),
    },
    "rk.whisper": {
        # WHISPER_WINDOW_S must match the seconds in the filename. rknn-lite
        # does not validate it: a mismatch reinterprets the buffer and the
        # transcript comes back as plausible nonsense.
        "base10": ("encoder/rk/whisper_encoder_base_10s.rknn", _WHISPER_DECODER_BASE),
        "base20": ("encoder/rk/whisper_encoder_base_20s.rknn", _WHISPER_DECODER_BASE),
    },
    "jetson.whisper_trt": {
        "base": ("encoder/jetson/enc_base_30s.onnx", _WHISPER_DECODER_BASE),
    },
}


def _ensure_whisper_artifacts(spec: str) -> None:
    """Download the Whisper assets for one encoder execution path (idempotent).

    ``WHISPER_VARIANT`` selects within a path (Hailo ``tiny``/``base``, RK
    ``base10``/``base20``). Files land under ``WHISPER_MODEL_DIR`` keeping their
    repo-relative layout, which is what the leaf's env values point at. Honors
    HF_ENDPOINT mirrors; the repo is overridable via ``WHISPER_HF_REPO``.
    """
    variants = _WHISPER_ENCODER_FILES.get(spec)
    if not variants:
        return
    default_variant = "base10" if spec == "rk.whisper" else "base"
    variant = os.environ.get("WHISPER_VARIANT", default_variant).lower()
    if variant not in variants:
        raise RuntimeError(
            f"WHISPER_VARIANT={variant!r} is not valid for {spec}; "
            f"choose one of {sorted(variants)}"
        )
    encoder, decoder_files = variants[variant]

    dest = os.environ.get("WHISPER_MODEL_DIR", "/opt/models/whisper")
    repo = os.environ.get("WHISPER_HF_REPO", "harvestsu/whisper-edge")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{repo}/resolve/main"

    for name in (encoder, *decoder_files, *_WHISPER_SHARED):
        path = os.path.join(dest, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info("Whisper asset OK: %s", name)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"{base}/{name}"
        logger.info("Downloading Whisper asset %s ...", url)
        tmp = path + ".part"
        try:
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
                     "--retry", "3", "-o", tmp, url],
                    check=True, timeout=1900,
                )
            else:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
                with urllib.request.urlopen(req, timeout=1800) as resp, open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            os.replace(tmp, path)
            logger.info("Whisper asset ready: %s (%d bytes)", name, os.path.getsize(path))
        except Exception as exc:
            logger.error("Failed to download Whisper asset %s: %s", name, exc)
            logger.error("Manually place %s under %s", name, dest)
            raise

    if spec == "jetson.whisper_trt":
        # The profile points WHISPER_ENCODER_PATH at the GENERATED plan; only
        # the ONNX ships. Without this the profile starts and then fails at
        # model load on a file nothing ever creates.
        onnx_path = os.path.join(dest, encoder)
        plan_path = os.environ.get("WHISPER_ENCODER_PATH") or os.path.join(
            dest, _WHISPER_TRT_PLAN
        )
        parent = os.path.dirname(plan_path)
        if parent:            # a bare filename has no parent; makedirs("") raises
            os.makedirs(parent, exist_ok=True)
        _build_whisper_trt_engine(onnx_path, plan_path)



# The Jetson path ships ONNX, not a .plan: TRT engines are version-specific, so
# the engine is built here against the host TensorRT that will run it.
_WHISPER_TRT_ONNX = "encoder/jetson/enc_base_30s.onnx"


def _whisper_engine_is_the_encoder(engine, plan_path: str, frames: int) -> None:
    """Reject an engine that is not this backend's encoder.

    ``TensorRTEncoder`` addresses tensors positionally — index 0 is the mel in,
    index 1 the hidden states out. Any same-version plan deserializes, so
    pointing WHISPER_ENCODER_PATH at, say, sensevoice.plan is accepted and then
    silently misinterpreted. Check the shape of the contract instead.
    """
    n = engine.num_io_tensors
    if n != 2:
        raise RuntimeError(
            f"{plan_path} has {n} I/O tensors; the Whisper encoder has 2 "
            f"(mel in, hidden states out) and they are addressed by position"
        )
    name = engine.get_tensor_name(0)
    shape = tuple(engine.get_tensor_shape(name))
    if len(shape) != 3 or shape[1] != N_MELS_WHISPER:
        raise RuntimeError(
            f"{plan_path} input {name!r} has shape {shape}; the Whisper encoder "
            f"takes [batch, {N_MELS_WHISPER}, frames]"
        )
    # The frame count too. 80 mel channels is not distinctive — a Vocos
    # vocoder profiled for 1x80x72..600 has the same rank and channel count and
    # would otherwise pass, only to be fed 1x80x3000 at the first utterance.
    if shape[2] not in (-1, frames):
        raise RuntimeError(
            f"{plan_path} input {name!r} takes {shape[2]} frames; this window "
            f"needs {frames}. That is a different model, or an engine built for "
            f"a different window."
        )


# Whisper's mel filterbank width; the encoder input's channel dimension.
N_MELS_WHISPER = 80


def _whisper_trt_build_spec(onnx_path: str) -> dict:
    """Everything that changes the artifact, recorded beside it.

    Mirrors the SenseVoice sidecar: a spec change rebuilds rather than silently
    serving a stale engine from a different TensorRT or a different precision.
    """
    return {
        "onnx_sha256": _file_sha256(onnx_path)[:16],
        "precision": "bf16",
        "shape": "1x80x3000",
        "workspace_gib": _env_int("WHISPER_TRT_WORKSPACE_GIB", 3),
    }


def _build_whisper_trt_engine(onnx_path: str, plan_path: str) -> None:
    """Build the Whisper encoder engine with the host-mounted TensorRT.

    **BF16, never FP16.** The fp16 build of this graph produces an engine whose
    output scores cosine 0.826 against onnxruntime — deterministic run to run,
    so it is kernel selection rather than a race — and it raises nothing: the
    decoder goes on emitting fluent English that drifts off-topic. bf16 keeps
    fp32's exponent range, scores 0.9996, and matches fp32's error rates while
    the encoder costs 12.5 ms against fp32's 39.1 ms. Setting both flags does
    NOT get a per-layer mix; TRT picks fp16 throughout and the engine comes out
    bit-identical to the pure fp16 one. See
    docs/perf/whisper-cross-device-20260827.md.

    Verify any engine with bench/perf/whisper/cmp_engine_precision.py before
    trusting it — the failure mode is invisible by inspection.
    """
    import tensorrt as trt

    spec = _whisper_trt_build_spec(onnx_path)
    info_path = plan_path + ".buildinfo.json"

    # A plan with no sidecar was supplied by hand — the documented escape hatch
    # for a TensorRT that cannot build BF16. Treating it as stale would rebuild
    # it, and on such a TensorRT that rebuild raises, making the escape hatch
    # unusable. Use it, and say that its precision is unverified.
    if (os.path.exists(plan_path) and os.path.getsize(plan_path) > 0
            and not os.path.exists(info_path)):
        probe_logger = trt.Logger(trt.Logger.ERROR)
        probe_runtime = trt.Runtime(probe_logger)   # must outlive the engine
        with open(plan_path, "rb") as fh:
            engine = probe_runtime.deserialize_cuda_engine(fh.read())
        if engine is None:
            raise RuntimeError(
                f"hand-supplied Whisper TRT engine {plan_path} does not "
                f"deserialize with TensorRT {trt.__version__}; engines are "
                f"version-specific"
            )
        # 100 frames per second of audio, from the shared geometry table.
        window_s, _cutoff = _WHISPER_GEOMETRY[("jetson.whisper_trt", "base")]
        _whisper_engine_is_the_encoder(engine, plan_path, int(window_s * 100))
        del engine, probe_runtime, probe_logger
        logger.warning(
            "Using hand-supplied Whisper TRT engine %s; it deserializes, but its "
            "PRECISION is not verified. An fp16 build of this graph fails "
            "silently — check it with bench/perf/whisper/cmp_engine_precision.py",
            plan_path,
        )
        return
    if os.path.exists(plan_path) and os.path.getsize(plan_path) > 0:
        try:
            with open(info_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            # Size too: a sidecar alone says the build INPUTS are unchanged,
            # not that the artifact survived. A plan truncated by a full disk
            # or an interrupted copy keeps its sidecar and would be trusted
            # until deserialization fails at model load.
            if (cached.get("spec") == spec
                    and cached.get("trt") == trt.__version__
                    and cached.get("plan_bytes") == os.path.getsize(plan_path)
                    and cached.get("plan_sha256") == _file_sha256(plan_path)):
                logger.info("Whisper TRT engine up to date: %s", plan_path)
                return
            logger.info("Whisper TRT engine stale (spec or TRT version changed); rebuilding")
        except (OSError, json.JSONDecodeError):
            logger.info("Whisper TRT engine has no usable buildinfo; rebuilding")

    logger.info(
        "Building Whisper TRT encoder (host TRT %s, bf16) from %s", trt.__version__, onnx_path
    )
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("  TRT parse error: %s", parser.get_error(i))
            raise RuntimeError(f"Whisper ONNX parse failed: {onnx_path!r}")

    # The encoder ONNX declares `batch_size` dynamic. TensorRT refuses to build
    # a network with a dynamic input unless an optimization profile pins it, so
    # without this the build does not merely underperform — it produces nothing.
    # Everything but batch is fixed by the compiled window, so min == opt == max.
    profile = builder.create_optimization_profile()
    pinned = 0
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if any(d < 0 for d in tensor.shape):
            shape = tuple(1 if d < 0 else d for d in tensor.shape)
            profile.set_shape(tensor.name, shape, shape, shape)
            pinned += 1
            logger.info("Whisper TRT: pinned dynamic input %s to %s", tensor.name, shape)

    config = builder.create_builder_config()
    # Count them ourselves. IOptimizationProfile exposes only set_shape /
    # get_shape / set_shape_input / get_shape_input / extra_memory_target —
    # verified against TensorRT 10.3 on the device. Reading `.num_inputs` raised
    # AttributeError before `add_optimization_profile` was ever reached, so the
    # build produced nothing at all.
    if pinned:
        config.add_optimization_profile(profile)
    if not hasattr(trt.BuilderFlag, "BF16"):
        raise RuntimeError(
            f"TensorRT {trt.__version__} has no BF16 builder flag. Building this "
            f"encoder in fp16 fails SILENTLY (cosine 0.826, fluent off-topic "
            f"output), so there is no safe fallback — upgrade TensorRT or supply "
            f"a prebuilt bf16 plan at {plan_path}"
        )
    config.set_flag(trt.BuilderFlag.BF16)
    ws_gib = int(spec["workspace_gib"])
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_gib << 30)
    except Exception:
        config.max_workspace_size = ws_gib << 30  # older TRT

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Whisper TRT build_serialized_network returned None")
    tmp = plan_path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(bytes(plan))
    os.replace(tmp, plan_path)
    with open(info_path, "w", encoding="utf-8") as fh:
        # Hash as well as size: a single flipped byte in a plan preserves both
        # its length and its sidecar, and was reported "up to date" until
        # deserialization failed at model load.
        json.dump({"trt": trt.__version__, "spec": spec,
                   "plan_bytes": os.path.getsize(plan_path),
                   "plan_sha256": _file_sha256(plan_path)}, fh, indent=2, sort_keys=True)
    logger.info(
        "Whisper TRT engine built: %s (%d bytes)", plan_path, os.path.getsize(plan_path)
    )


# SenseVoice on Jetson: the rescaled fixed ONNX (fp16-safe for Chinese) + decode
# assets; the engine is built on-device with the host-mounted TensorRT so it
# matches the runtime (TRT engines are version-specific — no universal .plan).
_SENSEVOICE_TRT_ONNX = "sense-voice-encoder.scaled.fixed.onnx"


def _ensure_sensevoice_trt_artifacts() -> None:
    """Provision the SenseVoice Jetson TRT backend (idempotent).

    Downloads the rescaled fixed ONNX + decode assets from HF into
    ``SENSEVOICE_TRT_MODEL_DIR`` and builds the fp16 TensorRT engine with the
    host-mounted TRT if it's missing. Honors HF_ENDPOINT; repo overridable via
    ``SENSEVOICE_RKNN_HF_REPO`` (shared with the RK assets).
    """
    dest = os.environ.get("SENSEVOICE_TRT_MODEL_DIR", "/opt/models/sensevoice-trt")
    repo = os.environ.get("SENSEVOICE_RKNN_HF_REPO", "harvestsu/sensevoice-rknn")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{repo}/resolve/main"

    os.makedirs(dest, exist_ok=True)
    for name in (_SENSEVOICE_TRT_ONNX, *_SENSEVOICE_RKNN_SHARED):
        path = os.path.join(dest, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info("SenseVoice TRT asset OK: %s", name)
            continue
        url = f"{base}/{name}"
        logger.info("Downloading SenseVoice TRT asset %s ...", url)
        tmp = path + ".part"
        try:
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
                     "--retry", "3", "-o", tmp, url],
                    check=True, timeout=1900,
                )
            else:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
                with urllib.request.urlopen(req, timeout=1800) as resp, open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            os.replace(tmp, path)
            logger.info("SenseVoice TRT asset ready: %s (%d bytes)", name, os.path.getsize(path))
        except Exception as exc:
            logger.error("Failed to download SenseVoice TRT asset %s: %s", name, exc)
            raise

    engine = os.environ.get("SENSEVOICE_TRT_ENGINE") or os.path.join(dest, "sensevoice.plan")
    onnx_path = os.path.join(dest, _SENSEVOICE_TRT_ONNX)
    spec = _sensevoice_build_spec(onnx_path)
    stale = _sensevoice_engine_staleness(engine, spec)
    if stale is None:
        logger.info("SenseVoice TRT engine present and current: %s", engine)
        return
    if os.path.exists(engine):
        logger.info("SenseVoice TRT engine rebuild — %s", stale)
    _build_sensevoice_trt_engine(onnx_path, engine, spec)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _file_sha256(path: str) -> str:
    """Content digest of ``path``; ``""`` when unreadable."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _sensevoice_build_spec(onnx_path: str) -> dict:
    """Everything that changes the produced ``.plan``, in one place.

    Add a knob here and it is automatically part of the cache key, so flipping
    it forces a rebuild instead of silently reusing an engine built with the
    old settings — the failure mode that makes a "just change a parameter"
    pipeline untrustworthy.
    """
    return {
        # Source identity. Name+size cannot tell two different models of the
        # same size apart, and a re-download that happened to match would leave
        # a stale plan looking current, so hash the contents. Cost is one read
        # of the ONNX at provisioning time; the alternative is serving an engine
        # built from a different model with nothing in the logs.
        "onnx": os.path.basename(onnx_path),
        "onnx_sha256": _file_sha256(onnx_path),
        # Precision. fp16 is required for speed but the ONNX must be the
        # activation-rescaled variant, else Chinese decodes to NaN.
        "fp16": _env_flag("SENSEVOICE_TRT_FP16", True),
        # Builder scratch (build-time only, not runtime memory).
        "workspace_gib": _env_int("SENSEVOICE_TRT_WORKSPACE_GIB", 3),
        # TRT 10 builder effort: higher = slower build, possibly faster engine.
        "opt_level": _env_int("SENSEVOICE_TRT_OPT_LEVEL", -1),
        # Fold the vocab argmax into the engine so D2H carries (1, T) int32
        # instead of (1, T, 25055) fp32 — measured 34.5 MB -> 1.4 KB, worth
        # ~8.8 ms per request. OFF by default: the runtime must be argmax-aware
        # to consume the changed output, so flipping this alone breaks decode.
        "argmax": _env_flag("SENSEVOICE_TRT_ARGMAX", False),
    }


def _trt_version_or_none() -> Optional[str]:
    """Installed TensorRT version, or ``None`` where TRT is not importable.

    A seam: engines are version-specific, so this drives the staleness check
    and tests substitute it to simulate an upgrade.
    """
    try:
        import tensorrt as trt
        return str(trt.__version__)
    except Exception:
        return None


def _sensevoice_engine_staleness(plan_path: str, spec: dict) -> Optional[str]:
    """``None`` when the cached engine matches ``spec``; else why it does not."""
    if not (os.path.exists(plan_path) and os.path.getsize(plan_path) > 0):
        return "no engine on disk"
    sidecar = plan_path + ".buildinfo.json"
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
    except (OSError, ValueError):
        # Engine from before build info existed: keep it rather than force a
        # multi-minute rebuild on upgrade, but only while the spec is default.
        return None if spec == _sensevoice_build_spec_defaults(spec) else (
            "engine predates build info and the build spec is non-default"
        )
    trt_version = _trt_version_or_none()
    if trt_version is None:
        # Cannot compare without TensorRT — and without it the rebuild could
        # not run either, so leave the engine alone rather than guess.
        logger.debug("TensorRT unavailable; skipping engine version check")
    elif prev.get("trt") != trt_version:
        return f"TensorRT changed: {prev.get('trt')} -> {trt_version}"
    changed = [
        f"{k}: {prev.get('spec', {}).get(k)!r} -> {v!r}"
        for k, v in spec.items()
        if prev.get("spec", {}).get(k) != v
    ]
    return ("build spec changed — " + ", ".join(changed)) if changed else None


def _sensevoice_build_spec_defaults(spec: dict) -> dict:
    """``spec`` as it would be with no env overrides (same onnx identity)."""
    defaults = {"fp16": True, "workspace_gib": 3, "opt_level": -1, "argmax": False}
    return {**spec, **defaults}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("%s is not an int; using %d", name, default)
        return default


def _append_argmax(trt, network) -> None:
    """Fold the vocab argmax into the engine.

    The encoder emits ``(1, T, V)`` CTC logits and the runtime only ever takes
    ``argmax(-1)`` of them, so shipping the full tensor back costs 34.5 MB of
    D2H per request to use ``T`` integers. A TopK(k=1) over the vocab axis
    moves that reduction onto the GPU.

    Callers must keep the runtime in step: the output becomes ``(1, T, 1)``
    int32 indices, so a backend still expecting logits will break. That is why
    ``SENSEVOICE_TRT_ARGMAX`` is off by default and recorded in the sidecar.
    """
    out = network.get_output(0)
    vocab_axis = len(out.shape) - 1
    topk = network.add_topk(out, trt.TopKOperation.MAX, 1, 1 << vocab_axis)
    if topk is None:
        raise RuntimeError("SenseVoice TRT: add_topk failed while folding argmax")
    indices = topk.get_output(1)
    indices.name = "encoder_argmax"
    network.unmark_output(out)
    network.mark_output(indices)
    logger.info(
        "SenseVoice TRT: folded argmax over axis %d — output %s -> %s int32",
        vocab_axis, tuple(out.shape), tuple(indices.shape),
    )


# Custom voice patches: replace unused speakers in voices.bin with custom voices.
# Each voice embedding is (510, 1, 256) float32 = 522240 bytes.
# Patches are stored in /opt/speech/voices/ (baked into Docker image).
_VOICE_PATCHES = {
    52: "af_cute.bin",  # replaces zm_yunyang (sid=52) with cute voice
}
_VOICE_BYTES = 510 * 1 * 256 * 4  # 522240


def _build_sensevoice_trt_engine(
    onnx_path: str, plan_path: str, spec: Optional[dict] = None
) -> None:
    """Build the SenseVoice engine from ONNX with the host-mounted TensorRT.

    Cached against ``spec`` (see ``_sensevoice_build_spec``): every knob that
    changes the artifact is recorded in a ``<plan>.buildinfo.json`` sidecar and
    compared on the next start, so changing one env var rebuilds rather than
    silently serving a stale engine. The host TRT matches the runtime, so the
    engine always deserializes.
    """
    import tensorrt as trt

    if spec is None:
        spec = _sensevoice_build_spec(onnx_path)
    logger.info(
        "Building SenseVoice TRT engine (host TRT %s) from %s — spec: %s",
        trt.__version__, onnx_path,
        ", ".join(f"{k}={v}" for k, v in sorted(spec.items())),
    )
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("  TRT parse error: %s", parser.get_error(i))
            raise RuntimeError(f"SenseVoice ONNX parse failed: {onnx_path!r}")
    if spec.get("argmax"):
        _append_argmax(trt, network)

    config = builder.create_builder_config()
    if spec.get("fp16", True):
        config.set_flag(trt.BuilderFlag.FP16)
    ws_gib = int(spec.get("workspace_gib", 3))
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_gib << 30)
    except Exception:
        config.max_workspace_size = ws_gib << 30  # older TRT
    opt_level = int(spec.get("opt_level", -1))
    if opt_level >= 0:
        try:
            config.builder_optimization_level = opt_level
        except Exception:
            logger.warning("This TensorRT has no builder_optimization_level; ignoring")

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("SenseVoice TRT build_serialized_network returned None")
    tmp = plan_path + ".part"
    with open(tmp, "wb") as f:
        f.write(bytes(plan))
    os.replace(tmp, plan_path)
    with open(plan_path + ".buildinfo.json", "w", encoding="utf-8") as fh:
        json.dump({"trt": trt.__version__, "spec": spec}, fh, indent=2, sort_keys=True)
    logger.info("SenseVoice TRT engine built: %s (%d bytes)", plan_path, os.path.getsize(plan_path))


# Custom voice patches: replace unused speakers in voices.bin with custom voices.
# Each voice embedding is (510, 1, 256) float32 = 522240 bytes.
# Patches are stored in /opt/speech/voices/ (baked into Docker image).
_VOICE_PATCHES = {
    52: "af_cute.bin",  # replaces zm_yunyang (sid=52) with cute voice
}
_VOICE_BYTES = 510 * 1 * 256 * 4  # 522240


def _patch_kokoro_voices(model_dir: str) -> None:
    """Patch voices.bin with custom voice embeddings if not already applied."""
    voices_bin = os.path.join(model_dir, "kokoro-multi-lang-v1_0", "voices.bin")
    if not os.path.isfile(voices_bin):
        return

    patch_dir = os.path.join(os.path.dirname(__file__), "..", "voices")
    marker = voices_bin + ".patched"

    if os.path.isfile(marker):
        return

    for sid, patch_file in _VOICE_PATCHES.items():
        patch_path = os.path.join(patch_dir, patch_file)
        if not os.path.isfile(patch_path):
            logger.warning("Voice patch %s not found, skipping", patch_path)
            continue
        with open(patch_path, "rb") as f:
            patch_data = f.read()
        if len(patch_data) != _VOICE_BYTES:
            logger.warning("Voice patch %s has wrong size %d, skipping", patch_file, len(patch_data))
            continue
        offset = sid * _VOICE_BYTES
        with open(voices_bin, "r+b") as f:
            f.seek(offset)
            f.write(patch_data)
        logger.info("Patched voices.bin sid=%d with %s", sid, patch_file)

    # Write marker so we don't re-patch on every startup
    with open(marker, "w") as f:
        f.write("patched\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    ensure_models(mode, model_dir)
