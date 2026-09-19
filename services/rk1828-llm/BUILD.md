# Building the RK1828 LLM service image

OpenAI-compatible LLM endpoint backed by the RK1828 PCIe NPU accelerator card.
The repo already had an LLM chat service for Jetson (`edge-llm-chat-service`);
this is the Rockchip counterpart, which did not exist.

The RK1828 uses **RKNN3 / RKLLM3 V1.0.4** — a different toolchain from the
RKNN2 in the voice image. The two cannot share a base image.

## What is in git and what is not

| | |
|---|---|
| In git | `rk1828_llm_server.py` (the OpenAI shim), `patches/Qwen3-cpp-main.cc` (the server-mode worker source), `artifacts.json`, `entrypoint.sh`, `Dockerfile` |
| **Not** in git | `deploy/rk1828-runtime/` — the compiled worker binary + `librknn3_api.so`. Same policy as `deploy/rk-runtime/`: a staged build artifact with a `MANIFEST.json` for provenance |
| **Never** anywhere | the model artifacts (3.2 GB). Pulled at runtime, see below |

## Step 1 — build the worker binary (on an RK1828 host)

The worker is a server-mode conversion of the RKNN3 model-zoo Qwen3 demo: init
once, then read framed requests on stdin and stream token frames on stdout. The
upstream demo is a one-shot CLI that takes the prompt as `argv`.

`patches/Qwen3-cpp-main.cc` is the converted source. Drop it over the model zoo's
copy and build **only** via the zoo's own script:

```bash
cd ~/rk1828/rknn3-model-zoo
cp <repo>/services/rk1828-llm/patches/Qwen3-cpp-main.cc examples/Qwen3/cpp/main.cc
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3
```

**Never call cmake/make directly** — a bare cmake build produces ABI-incompatible
artifacts.

Protocol implemented by the converted worker (do not redesign it):

* argv: `<model_dir> [--core-mask <hex>] [--max-context <n>] [--device-id <id>] -`
  — the trailing `-` selects server mode; the original 6-arg one-shot path still works.
* stderr: `READY 1` handshake plus all diagnostics.
* stdin: one request per line, `<max_new_tokens>\t<escaped prompt>`.
* stdout: `[uint32 LE len][utf8 token]` per token, `[uint32 LE 0xFFFFFFFE]` as the
  per-request EOS sentinel.

Three implementation details that are load-bearing:

1. **stdout must carry only protocol frames.** `dup(STDOUT)→frame_fd` then
   `dup2(stderr, stdout)` **before** model init — the RKNN runtime prints during
   init, and any of that text on the frame channel desyncs the reader into a
   multi-gigabyte read.
2. **`rknn3_session_clear_kvcache(..., RKNN3_KVCACHE_CLEAR_ALL)` after EVERY
   request, including error and empty-prompt paths.** A skipped clear leaves dirty
   KV that accumulates until it overflows `max_context` and aborts at runtime.
3. **Do not hold the worker lock inside the SSE generator.** If a client
   disconnects mid-stream the generator is abandoned still holding the lock with
   unread frames queued; the next request then blocks forever, or desyncs onto the
   previous request's tokens. A dedicated thread must own the lock and always
   drain through the EOS sentinel regardless of whether the HTTP consumer is
   still there. (Same class as the `wanted 1852143441 bytes` desync noted in
   rkvoice-stream's `docs/rk1828-package-integration.md` §12.2.)

## Step 2 — stage the runtime

Copy the binary and **the WHOLE `lib/` directory** from the device:

```bash
mkdir -p deploy/rk1828-runtime/lib
# from install/rk3588_linux_aarch64/rknn_Qwen3_demo/ on the build device:
#   rknn_qwen3_demo        -> deploy/rk1828-runtime/
#   lib/*                  -> deploy/rk1828-runtime/lib/     (ALL of it, see below)
```

Layout matters: the binary's rpath is `$ORIGIN/lib`, so the libraries must sit in a
`lib/` directory **next to** the binary.

### ⚠️ Copy all three libs — and `ldd` will NOT catch it if you don't

`lib/` holds three files, and it is not obvious which matter:

| | |
|---|---|
| `librknn3_api.so` (~56 KB) | **only a dispatch shim** |
| `librknn3_api_rkcp.so` (~8.6 MB) | **the actual RKNN3 implementation** |
| `librga.so` (~197 KB) | graphics/buffer helper |

`librknn3_api.so` **`dlopen()`s** `librknn3_api_rkcp.so` at runtime, searching
`./`, `./lib/`, `/usr/local/lib`, `/usr/lib`. Because that is a dlopen and not a
link-time dependency:

* `ldd rknn_qwen3_demo` resolves **cleanly** with the real implementation absent —
  so an `ldd`-based image check passes and tells you nothing;
* the only symptom is `rknn_init fail ret=-1` at model load, which reads like a
  card, firmware or model problem rather than a missing file.

This exact mistake was made once already: staging copied `librknn3_api.so` alone,
`ldd` looked perfect, and the failure surfaced only as a failed model init — after
burning part of the retry budget on a card that was in fact fine. Verify with
`strings lib/librknn3_api.so | grep rkcp` if you ever doubt which is which.

The Dockerfile does a whole-directory `COPY deploy/rk1828-runtime/lib/`, so nothing
needs changing there — the risk is purely in what you stage.

Then write `MANIFEST.json` recording sizes, md5s and sha256s of every staged file,
the SDK version (V1.0.4) and the sha256 of the `main.cc` the worker was built from.
Unlike `deploy/rk-runtime/`, this manifest IS committed — it is the only record of
what the image expects, so a mismatched staging can be caught rather than shipped.

### Build context

`.dockerignore` excludes `deploy/` wholesale and re-includes specific
subdirectories, so a NEW staged directory is invisible to the build until it is
allowlisted. `deploy/rk1828-runtime/` (and `agent/README.md`, which the agent image
COPYs) are already listed; if you add another staged directory, add it there too or
the build fails with a confusing "not found" on a file that plainly exists.

## Step 3 — build the image

Build context is the **repo root**, because the runtime is staged under `deploy/`:

```bash
docker build -f services/rk1828-llm/Dockerfile -t edge-llm-rk1828:$(date +%Y%m%d) .
```

Native arm64. Resulting image is ~200 MB (no models).

## Runtime

Models are pulled on first start into `RK1828_MODEL_DIR` — mount a named volume
so a container replacement does not re-download 3.2 GB.

| Env | Default | Notes |
|---|---|---|
| `RK1828_ARTIFACT_AUTO_DOWNLOAD` | `1` | `0` = expect models already present |
| `RK1828_ARTIFACT_REPO_ID` | `harvestsu/seeed-local-voice-rk-artifacts` | |
| `RK1828_ARTIFACT_PREFIX` | `rk1828/opt/llm/qwen3-4b` | matches the repo's SoC-prefixed layout |
| `HF_ENDPOINT` | `https://huggingface.co` | set a mirror behind the firewall; mirrors may lag |
| `RK1828_MAX_CONTEXT` | `8192` | verified; a runtime parameter, no re-export needed |
| `RK1828_CORE_MASK` | `ff` | 8 cores |
| `RK1828_PORT` | `1828` | |
| `RK1828_HOST_LIB_DIR` | `/opt/rk1828/host-lib` | where the host's `/usr/lib` is mounted; see *Runtime alignment* |
| `RK1828_PREFER_HOST_RUNTIME` | `1` | `0` keeps the bundled client lib even when the host has a different one |

The entrypoint verifies file SIZES, not just existence — a half-finished download
would otherwise be treated as present and fail model init with something far less
obvious. It also refuses to start on an incomplete set, because the four files are
a matched export and a mismatched pair surfaces as a firmware-level `ACK_FAIL`
that looks like a hardware fault.

## Runtime alignment (host ↔ image ↔ model)

Three things have versions and only two of the three pairings are free:

| pairing | rule | evidence |
|---|---|---|
| runtime ↔ model | a **higher** runtime loads a **lower**-version model, so bumping the client does not require re-exporting the 3.2 GB artifacts | Rockchip, 2026-09-18 |
| client lib ↔ host proxy + EP firmware | must be the **same generation**; they ship in one host package | 2026-09-17 failure below |
| model ↔ model | the four artifact files are one matched export, all-or-nothing | `entrypoint.sh` size/set check |

So the image cannot pin the client lib and be portable: the host's copy is the
one in lockstep with the proxy and the firmware. `entrypoint.sh` therefore copies
`librknn3_api.so` + `librknn3_api_rkcp.so` from `RK1828_HOST_LIB_DIR` over the
bundled pair at startup, and both composes mount the host's `/usr/lib` there
read-only. The bundled pair stays as the fallback for a host that does not expose
it. The whole dir is mounted rather than the two files because a bind mount of a
file the host does not have makes docker **create a directory** at that host path.

### The failure this prevents (2026-09-17, recomputer-rk3588-devkit)

| | radxa (works) | devkit (hung) |
|---|---|---|
| host proxy | `Transfer version 1.0.4 (8ab764c@2026-05-13)` | `1.1.0 (13b774e@2026-08-13)` |
| host `librknn3_api_rkcp.so` | 8990352 B, md5 `79dad96c…` — **byte-identical to the image's bundled copy** | 9078912 B |
| install | manual, no dpkg record | apt `rknn3-rk182x-m2` 1.1.0 |
| kernel / OS | 6.1.84-8-rk2410 / Debian 12 | 6.1.172-vendor-seeed-rk3588 / Armbian 26.04 |

Diagnostic signature of a skew — worth recognising, because nothing logs it:

* every start attempt times out after the **same** duration (measured: 179.6 s,
  spread < 60 ms across five attempts). A genuinely slow load gets faster on
  attempt 2 once the page cache is warm.
* the worker's last line is `[server] --> init qwen3 llm model`.
* the worker sits in `read()` on fd 5 = `socket:[…]` to the abstract socket
  `@transfer_proxy3`, kernel stack `unix_stream_recvmsg`, both queues empty.
* RSS ~50 MB and ~0.5 s of CPU — the 3.2 GB was never read.
* `rknn-smi info -t memory` shows the EP untouched (5087 of 5120 MB free).

Storage was ruled out on that box, so do not start there: `dd` of the 2.43 GB
weight file on the eMMC/SD rootfs gave **44 MB/s** cold, i.e. ~74 s for the whole
3.22 GB set — 106 s of headroom inside the 180 s ceiling.

## Host prerequisites — the image is NOT self-contained

The card's kernel driver and firmware live on the host:

* `pcie_rkep` module loaded. **It does not persist across reboot** — make it
  durable on the host.
* `rknn3.service` active; it reflashes the EP firmware at host boot.

Acceptance before starting the container:

```bash
lspci | grep 182a                      # expect 0001:11:00.0 ... Device 182a
systemctl is-active rknn3.service      # expect active
ls /dev/pcie-rkep-*                    # expect the char device
```

The container needs `privileged: true` and the char device. The entrypoint fails
fast with a pointed message if the device is not visible.

## What the RKNN3 runtime actually supports (measured 2026-07-31)

The header promises more than the runtime delivers, and the gap is not obvious
from the docs. Everything below was measured on this device, not inferred.

### Function tools — use the runtime, not a hand-written preamble

`rknn3_session_set_function_tools(session, tools_json)` works (`ret=0`). The
runtime renders the schema through the **model's own Jinja chat template**
(minja, read from the GGUF), whose `{%- if tools %}` branch emits the canonical
Qwen3 tool preamble into a real **system** block:

```
tools not registered : prefill  15 tokens, model answers in prose, calls nothing
tools registered     : prefill 239 tokens, model emits a well-formed <tool_call>
```

The shim used to hand-write a copy of that preamble into the user turn. It
scored 12/12 at temperature 0 — because it reproduced the template's own wording
by hand — but the template is the model's and does not need keeping in sync.

Note the model still emits the call as **text** in the token stream either way,
so the shim's `ToolCallSplitter` is required regardless.

### KV prefix reuse — free, as long as you stop clearing

With `keep_history=0` the runtime maintains an automatic **prefix cache**: the
next request reuses whatever leading tokens it shares with the last one. Since
every request repeats the same system block and tool schema, that prefix is most
of the prompt.

The worker used to call `clear_kvcache(CLEAR_ALL)` after every request, which
threw that away. Measured end-to-end through the HTTP shim, six tool calls:

| | first call | steady state |
|---|---|---|
| clearing every request | 509 ms | 503–512 ms |
| prefix reuse (current) | 679 ms | **375–382 ms** |

~127 ms saved per tool call; the first call costs ~170 ms more because
registering tools re-renders the template. Break-even after two calls.

Verified safe over 12 consecutive **differing** requests: arguments always
tracked the current request (a prefix cache is not conversation history, so
nothing bleeds between requests), prefill stayed flat at 208–209 tokens — it
does not accumulate, so there is no drift toward `max_context` — and no request
returned empty. `RK1828_KV_REUSE=0` restores the old unconditional clear as an
escape hatch.

The older warning that the unconditional clear was load-bearing against dirty-KV
overflow was observed with `keep_history=1`, where turns genuinely accumulate. It
does not apply to the stateless path.

### Three things that do NOT work

| | |
|---|---|
| `rknn3_session_run` with more than one input | Rejected: `RKLLMSession: Only support one LLM input!`. The parameter is an array and `rknn3_llm_input.role` documents a `"tool"` role, but a conversation cannot be submitted in one call — history must be flattened into a single turn, or replayed turn by turn with `keep_history=1` |
| `RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT` | **Silently corrupts the session** once tools are registered: the next generation returns empty, and the one after that has the model reciting the tool preamble back as its answer. Not used, and not exposed by the worker protocol |
| `rknn3_session_set_chat_template(s, sys, NULL, NULL)` | `ret=-2, invalid arguments!` — prefix/postfix cannot be NULL. It is also the *legacy* simple-template path and looks mutually exclusive with the Jinja/function-tools path, so it is not needed |

### Multi-turn

Works, and always has. `keep_history=1` gives true incremental multi-turn
(verified: a fact stated in turn 1 is recalled in turn 3 without resending it),
but it makes the session **stateful**, which one shared EP cannot do safely when
more than one conversation is in flight.

The shipped path stays stateless: the caller sends the full `messages[]` every
turn, the shim flattens it into one prompt, and the prefix cache absorbs most of
the repeated cost. Multi-turn recall and cross-tool-turn recall are both verified
this way.

## Single-EP exclusivity

The card has ONE ~5 GB context, so **large models are mutually exclusive**.
Qwen3-4B at 8192 tokens uses an estimated ~3,640 MB of 5,120 MB (weights 2,432 MB
plus KV 8192 × 144 KB = 1,208 MB). Consequences:

* An RK1828-hosted TTS cannot run at the same time as this. That is why the voice
  image runs TTS on the RK3588's own NPU.
* EP memory observability **depends on the runtime generation**: `rknn-smi` is
  non-functional on V1.0.4 (radxa: fails as root, fails when the EP is idle —
  suspected host/EP firmware skew, `rc_cc_version=30301` vs `ep_cc_version=30201`)
  but works on the 1.1.0 package (devkit, 2026-09-18: `info -t memory` reports
  `Total DDR 5120.00 MB / Free 5087.34`, confirming the ~5 GB figure above).
  Where it does not work, model-load success is the only signal — and **failed
  loads degrade the EP from 8 cores to 4**, so do not probe by trial and error.
* **Never run `rknn-smi reset`** — it can wedge the card into a boot state that a
  host reboot may not recover, and the card has its own 12 V supply so it does not
  power-cycle with the host.
