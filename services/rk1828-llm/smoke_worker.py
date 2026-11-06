"""Dependency-free standalone smoke of the rknn_qwen3_demo server mode."""
import os, struct, subprocess, sys, threading, time

BIN = os.environ.get(
    "RK1828_BIN",
    "/opt/rk1828/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_demo/rknn_qwen3_demo",
)
MODEL = os.environ.get(
    "RK1828_MODEL",
    "/opt/rk1828/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_demo/model",
)
EOS = 0xFFFFFFFE

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = os.path.join(os.path.dirname(BIN), "lib") + ":/lib"
p = subprocess.Popen([BIN, MODEL, "--core-mask", "ff", "--max-context", "2048", "-"],
                     cwd=os.path.dirname(BIN), stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, bufsize=0)

ready = threading.Event()
errlines = []

def drain():
    for raw in iter(p.stderr.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        errlines.append(line)
        if "ready" in line.lower():
            ready.set()

threading.Thread(target=drain, daemon=True).start()
t0 = time.time()
if not ready.wait(180):
    print("READY TIMEOUT rc=", p.poll())
    print("\n".join(errlines[-20:]))
    sys.exit(1)
print("READY after %.1fs" % (time.time() - t0))

def read_exact(n):
    b = b""
    while len(b) < n:
        c = p.stdout.read(n - len(b))
        if not c:
            raise RuntimeError("EOF rc=%s tail=%s" % (p.poll(), errlines[-8:]))
        b += c
    return b

for i, prompt in enumerate(["用一句话介绍北京。", "What is 17 times 23? Answer briefly."]):
    print("--- request %d: %s" % (i, prompt))
    t = time.time()
    p.stdin.write(("128\t" + prompt.replace("\\", "\\\\").replace("\n", "\\n") + "\n").encode())
    p.stdin.flush()
    ntok, first, out = 0, None, b""
    while True:
        (ln,) = struct.unpack("<I", read_exact(4))
        if ln == EOS:
            break
        assert ln <= 8 * 1024 * 1024, "desync len=%d" % ln
        out += read_exact(ln)
        ntok += 1
        if first is None:
            first = time.time() - t
    print("tokens=%d ttft=%.3fs total=%.3fs" % (ntok, first or -1, time.time() - t))
    print("TEXT:", out.decode("utf-8", "replace"))

p.stdin.close()
print("exit rc=", p.wait(timeout=30))
print("STDERR TAIL:")
print("\n".join(errlines[-25:]))
