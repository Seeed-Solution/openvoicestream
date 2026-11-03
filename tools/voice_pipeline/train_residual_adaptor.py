import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time, requests, numba, librosa, csv
import numpy as np
import soundfile as sf
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
import kaldi_native_fbank as knf

sys.path.insert(0, ".")
sys.path.insert(0, "third_party/rkvoice-stream")
from rkvoice_stream.backends.tts.matcha import RKNNMatchaVocoder, parse_tokens_file

MEL_MEAN = -2.9521978
MEL_SIGMA = 5.446792
SAMPLE_RATE = 16000

print("[*] Initializing G2P and Vocoder...")
tokens_path = "models/matcha-icefall-zh-en/tokens.txt"
lexicon_path = "models/matcha-icefall-zh-en/lexicon.txt"
espeak_data = "models/matcha-icefall-zh-en/espeak-ng-data"
base_m6_path = "models/matcha-icefall-zh-en/model-steps-6.onnx"
vocos_path = "models/matcha-icefall-zh-en/vocos-16khz-univ.onnx"

vocoder = RKNNMatchaVocoder("", "", lexicon_path, tokens_path, espeak_data)
vocoder._token_to_id = parse_tokens_file(tokens_path)
vocoder._lexicon = {}
with open(lexicon_path, "r", encoding="utf-8") as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2: vocoder._lexicon[p[0]] = p[1:]

base_sess = ort.InferenceSession(base_m6_path, providers=["CPUExecutionProvider"])
vocos_sess = ort.InferenceSession(vocos_path, providers=["CPUExecutionProvider"])

stft_cfg = knf.StftConfig(n_fft=1024, hop_length=256, win_length=1024, window_type="hann", center=True, pad_mode="reflect", normalized=False)
istft = knf.IStft(stft_cfg)
def mel2wav(mel):
    mag, x, y = vocos_sess.run(["mag", "x", "y"], {"mels": mel})
    res = knf.StftResult(real=(mag*x)[0].transpose().reshape(-1).tolist(), imag=(mag*y)[0].transpose().reshape(-1).tolist(), num_frames=mag.shape[2])
    return np.array(istft(res))

# 1. Load dataset metadata
pairs = []
with open("voices/minimax_shaonv/metadata.csv", "r", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="|"):
        if len(row) >= 2:
            p = os.path.join("voices/minimax_shaonv/wavs", row[0].strip())
            if os.path.exists(p): pairs.append((p, row[1].strip()))

print(f"[*] Found {len(pairs)} training pairs. Extracting & DTW aligning...")
t0 = time.time()
aligned_pairs = []

for idx, (wav_path, text) in enumerate(pairs):
    t_ids = vocoder.text_to_tokens(text)
    if len(t_ids) == 0: continue
    toks = np.array([t_ids], dtype=np.int64)
    xlen = np.array([len(t_ids)], dtype=np.int64)

    # Base synthesis mel
    mel_base = base_sess.run(["mel"], {
        "x": toks, "x_length": xlen,
        "noise_scale": np.array([0.0], dtype=np.float32),
        "length_scale": np.array([1.0], dtype=np.float32)
    })[0][0] # [80, T_b]

    # Teacher MiniMax audio mel
    y_mm, _ = librosa.load(wav_path, sr=16000)
    y_mm = y_mm * (0.95 / np.max(np.abs(y_mm))) # peak normalize to 0.95
    mel_mm = librosa.feature.melspectrogram(
        y=y_mm, sr=16000, n_fft=1024, hop_length=256, win_length=1024,
        window="hann", n_mels=80, fmin=0.0, fmax=8000.0, power=1.0, center=True, pad_mode="reflect"
    )
    log_mel_mm = np.log(np.maximum(mel_mm, 1e-10)) # [80, T_mm]

    # DTW alignment between base mel and teacher mel
    # Normalize for cosine DTW
    nb = mel_base / (np.linalg.norm(mel_base, axis=0, keepdims=True) + 1e-6)
    nm = log_mel_mm / (np.linalg.norm(log_mel_mm, axis=0, keepdims=True) + 1e-6)
    _, wp = librosa.sequence.dtw(C=1.0 - (nb.T @ nm))

    # wp has shape [path_len, 2]
    # Filter out pure silence frames on both sides (< -15 dB)
    sub_base = mel_base[:, wp[:, 0]]
    sub_mm = log_mel_mm[:, wp[:, 1]]

    aligned_pairs.append((
        torch.from_numpy(mel_base).float(), # [80, T_b]
        torch.from_numpy(sub_base).float(), # [80, P]
        torch.from_numpy(sub_mm).float()    # [80, P]
    ))

print(f"[+] Processed {len(aligned_pairs)} pairs in {time.time() - t0:.2f}s.")

# 2. Define Timbre Residual Adaptor
class TimbreAdaptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(80, 128, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(128, 80, kernel_size=3, padding=1)
        # Initialize second conv with near-zero weights for smooth identity start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return x + res

adaptor = TimbreAdaptor()
optimizer = optim.AdamW(adaptor.parameters(), lr=2e-3, weight_decay=1e-5)
criterion = nn.SmoothL1Loss()

print("\n[*] Training Timbre Adaptor on aligned speech frames...")
adaptor.train()
for epoch in range(1, 21):
    total_loss = 0.0
    for _, sub_b, sub_m in aligned_pairs:
        # sub_b: [80, P], sub_m: [80, P]
        inp = sub_b.unsqueeze(0) # [1, 80, P]
        tgt = sub_m.unsqueeze(0) # [1, 80, P]

        optimizer.zero_grad()
        pred = adaptor(inp)

        # Spectral shape + magnitude loss
        l1_loss = criterion(pred, tgt)
        cos_loss = 1.0 - torch.cosine_similarity(pred, tgt, dim=1).mean()
        loss = l1_loss + 0.5 * cos_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:02d}/20] Loss: {total_loss / len(aligned_pairs):.4f}")

adaptor.eval()

# 3. Export adapted ONNX graph
print("\n[*] Integrating Timbre Adaptor into ONNX model...")
m_base = onnx.load(base_m6_path, load_external_data=False)

# Rename old output 'mel' to 'mel_pre_adapt'
for out in m_base.graph.output:
    if out.name == "mel":
        out.name = "mel_adapted"

for node in m_base.graph.node:
    for idx, o in enumerate(node.output):
        if o == "mel":
            node.output[idx] = "mel_pre_adapt"

# Extract weights from PyTorch adaptor
sd = adaptor.state_dict()
w1 = sd["conv1.weight"].numpy() # [128, 80, 3]
b1 = sd["conv1.bias"].numpy()   # [128]
w2 = sd["conv2.weight"].numpy() # [80, 128, 3]
b2 = sd["conv2.bias"].numpy()   # [80]

w1_init = numpy_helper.from_array(w1, name="adaptor.conv1.weight")
b1_init = numpy_helper.from_array(b1, name="adaptor.conv1.bias")
w2_init = numpy_helper.from_array(w2, name="adaptor.conv2.weight")
b2_init = numpy_helper.from_array(b2, name="adaptor.conv2.bias")

m_base.graph.initializer.extend([w1_init, b1_init, w2_init, b2_init])

# Add nodes: Conv1 -> Sigmoid -> Mul(SiLU) -> Conv2 -> Add(Residual) -> Output 'mel'
conv1_node = helper.make_node(
    "Conv",
    inputs=["mel_pre_adapt", "adaptor.conv1.weight", "adaptor.conv1.bias"],
    outputs=["adaptor.conv1_out"],
    pads=[1, 1],
    name="adaptor.conv1"
)
# SiLU(x) = x * Sigmoid(x)
sig_node = helper.make_node(
    "Sigmoid",
    inputs=["adaptor.conv1_out"],
    outputs=["adaptor.sig_out"],
    name="adaptor.sig"
)
silu_node = helper.make_node(
    "Mul",
    inputs=["adaptor.conv1_out", "adaptor.sig_out"],
    outputs=["adaptor.silu_out"],
    name="adaptor.silu"
)
conv2_node = helper.make_node(
    "Conv",
    inputs=["adaptor.silu_out", "adaptor.conv2.weight", "adaptor.conv2.bias"],
    outputs=["adaptor.conv2_out"],
    pads=[1, 1],
    name="adaptor.conv2"
)
add_node = helper.make_node(
    "Add",
    inputs=["mel_pre_adapt", "adaptor.conv2_out"],
    outputs=["mel"],
    name="adaptor.residual_add"
)

# Set final output
m_base.graph.node.extend([conv1_node, sig_node, silu_node, conv2_node, add_node])
for out in m_base.graph.output:
    if out.name == "mel_adapted":
        out.name = "mel"

output_onnx_path = "voices/minimax_shaonv/matcha_shaonv_step6.onnx"
onnx.save(m_base, output_onnx_path)
print(f"[+] Saved adapted ONNX: {output_onnx_path}")

# 4. Synthesize test cases and verify with LIVE ASR
test_sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
fleet_bin = os.path.expanduser("~/.rpty/bin/fleet")

test_cases = [
    ("minimax_shaonv_test01_bilingual.wav", "你好，我是 Seeed Studio 智能语音助手。Hello, welcome to the future.", "zh"),
    ("minimax_shaonv_test02_english.wav", "Welcome to Seeed Studio! We make open-source hardware accessible everywhere.", "en"),
]

print("\n" + "=" * 60)
print("=== LIVE ASR TRANSCRIPTION VERIFICATION ===")
print("=" * 60)

for fname, text, lang in test_cases:
    t_ids = vocoder.text_to_tokens(text)
    fixed_ids = []
    for i in range(len(t_ids)):
        fixed_ids.append(t_ids[i])
        if t_ids[i] == 1624 and i + 1 < len(t_ids) and t_ids[i+1] == 48:
            fixed_ids.append(1) # insert space token 1

    toks = np.array([fixed_ids], dtype=np.int64)
    xlen = np.array([len(fixed_ids)], dtype=np.int64)

    mel = test_sess.run(["mel"], {
        "x": toks, "x_length": xlen,
        "noise_scale": np.array([0.33], dtype=np.float32),
        "length_scale": np.array([1.0], dtype=np.float32)
    })[0]

    wav = mel2wav(mel)
    peak = np.max(np.abs(wav))
    if peak > 0: wav = wav * (0.90 / peak)

    out_path = f"voices/samples/{fname}"
    sf.write(out_path, wav, SAMPLE_RATE)
    rms = np.sqrt(np.mean(wav**2))
    dur = len(wav) / SAMPLE_RATE

    # Call local ASR
    with open(out_path, "rb") as f:
        resp = requests.post(f"http://127.0.0.1:8621/asr?language={lang}", files={"file": (fname, f, "audio/wav")})
        asr_txt = resp.json().get("text", "")

    print(f"\n[Case] {fname} ({dur:.2f}s, RMS={rms:.4f}, Peak={np.max(np.abs(wav)):.4f})")
    print(f"  Target Text: {text}")
    print(f"  ASR Result:  {asr_txt}")

    # Push to macbook
    os.system(f"{fleet_bin} push macbook {out_path} /Users/harvest/Downloads/{fname}")
    print(f"  [+] Pushed to MacBook: ~/Downloads/{fname}")
