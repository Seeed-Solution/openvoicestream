# OVS Kokoro ConvOnly productization evidence

## Source boundary

- Base: `c97126caad186423c8b78913bd22b42c92c7a2a2`
- Branch: `feat/kokoro-rk-productization`
- Worktree: `/home/harvest/project/ovs-kokoro-productization`
- The RKVoice submodule pointer is `32b4694e5946eb8bed63db6ed8116aa4b146aa94`
  (`rkvoice-stream==0.2.0`) and must be initialized before the source build.
- The integration commits are published on GitHub main. A clean clone fetched
  the pinned RKVoice submodule and verified version `0.2.0` plus the
  `kokoro_convonly` factory selection.

## Implemented surface

- OVS finite request cancellation, transport headers, and
  executor ownership in `server/core/api_execution.py`, `server/main.py`, and
  `server/core/tts_speakers.py`.
- Production RK3576/RK3588 ConvOnly profiles with model IDs and fixed manifest
  SHA values.
- explicit platform overlays, using one registry tag and
  read-only model/JA mounts. The overlay clears inherited ASR and TTS backend
  selections.
- Clean source clone RKVoice wheel build stage using `uv build`; no application wheel
  is copied from `deploy/wheels` by the ConvOnly target.
- Package locks, package checker, dictionary policy test, protocol tests, and
  English/Chinese README plus deployment documentation.

## Verification

```text
python3 -m py_compile server/core/api_execution.py server/core/tts_speakers.py server/main.py deploy/docker/check_kokoro_packages.py
PASS

docker compose -f deploy/docker-compose.rk.yml -f deploy/docker-compose.kokoro-convonly-rk3576.yml config
PASS; image=rk-20260903.10; ASR_BACKEND=disabled; TTS_BACKEND=kokoro_convonly

OVS_PROFILE=rk3588-kokoro-convonly RK_PLATFORM=rk3588 \
KOKORO_CONVONLY_MANIFEST_SHA256=83733c717e0ce5b76ac1295e4827cf3ad2e111955259e9d670897e100fabeb6e \
OVS_TTS_MODEL_ID=kokoro-convonly-v1_0-rk3588 \
docker compose -f deploy/docker-compose.rk3588.yml -f deploy/docker-compose.kokoro-convonly-rk3588.yml config
PASS; image=rk-20260903.10; ASR_BACKEND=disabled; TTS_BACKEND=kokoro_convonly; RK_PLATFORM=rk3588

uv run pytest -q ...
BLOCKED: pytest is not installed; uv temporary dependency install could not resolve
pypi.tuna.tsinghua.edu.cn because DNS/network access is unavailable in this environment.
```

## Release inputs and exclusions

The published HF repository is `harvestsu/seeed-local-voice-rk-artifacts` with
paths `rk3576/kokoro-convonly-v1_0/`, `rk3588/kokoro-convonly-v1_0/`, and
`resources/ja/unidic-lite-1.0.8/`. The published HF revision is
`3f8d58c8446ec4b18891624ad4ae4ce75e0f3d3e`; the registry digest is
`sha256:fdc480da30610f46075f41a8bf95be5774427a98d3e77c69272cdec1226593c1`. Model manifest SHA
values are the accepted RK3576 and RK3588 bundle hashes in the profiles.

Excluded from this worktree are RKVoice source changes, model/audio artifacts,
`deploy/wheels`, benchmark/experiment files, dated specs, and canary/Long32 changes.

`rkvoice-stream==0.2.0` and `voxedge==0.0.13a0` are published on PyPI. The
Rockchip runtime library directory and model bundles are explicit external build/deploy inputs.
The current worktree documents finite cancellation. The qualified `.10` image
continues to return one completed audio chunk.
