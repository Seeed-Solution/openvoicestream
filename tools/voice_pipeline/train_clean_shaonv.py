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

print("[*] Initializing G2P and Models...")
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

pairs = []
with open("voices/minimax_shaonv/metadata.csv", "r", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="|"):
        if len(row) >= 2:
            p = os.path.join("voices/minimax_shaonv/wavs", row[0].strip())
            if os.path.exists(p): pairs.append((p, row[1].strip()))

print(f"[*] Processing {len(pairs)} pairs with Energy Matching + DTW...")
t0 = time.time()
training_samples = []

for wav_path, text in pairs:
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
    y_mm = y_mm * (0.95 / np.max(np.abs(y_mm)))
    mel_mm = librosa.feature.melspectrogram(
        y=y_mm, sr=16000, n_fft=1024, hop_length=256, win_length=1024,
        window="hann", n_mels=80, fmin=0.0, fmax=8000.0, power=1.0, center=True, pad_mode="reflect"
    )
    log_mel_mm = np.log(np.maximum(mel_mm, 1e-10)) # [80, T_mm]

    # Filter speech frames
    speech_mask_b = mel_base.mean(axis=0) > -12.0
    speech_mask_mm = log_mel_mm.mean(axis=0) > -12.0
    if np.sum(speech_mask_b) < 15 or np.sum(speech_mask_mm) < 15:
        continue

    # 1. ENERGY MATCHING: Align MiniMax speech energy to Base speech energy!
    mean_energy_b = np.mean(mel_base[:, speech_mask_b])
    mean_energy_mm = np.mean(log_mel_mm[:, speech_mask_mm])
    energy_offset = mean_energy_b - mean_energy_mm
    log_mel_mm_calibrated = log_mel_mm + energy_offset

    # DTW alignment on speech frames only
    sub_b = mel_base[:, speech_mask_b]
    sub_mm = log_mel_mm_calibrated[:, speech_mask_mm]

    nb = sub_b / (np.linalg.norm(sub_b, axis=0, keepdims=True) + 1e-6)
    nm = sub_mm / (np.linalg.norm(sub_mm, axis=0, keepdims=True) + 1e-6)
    _, wp = librosa.sequence.dtw(C=1.0 - (nb.T @ nm))

    aligned_b = sub_b[:, wp[:, 0]]
    aligned_mm = sub_mm[:, wp[:, 1]]

    training_samples.append((
        torch.from_numpy(aligned_b).float(),
        torch.from_numpy(aligned_mm).float()
    ))

print(f"[+] Aligned {len(training_samples)} pairs in {time.time() - t0:.2f}s.")

# 2. Residual Adaptor without biases (strictly preserves zero/identity when not active)
class CleanTimbreAdaptor(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: 80 -> 128 (no bias)
        self.conv1 = nn.Conv1d(80, 128, kernel_size=3, padding=1, bias=False)
        self.act = nn.SiLU()
        # Conv2: 128 -> 80 (no bias)
        self.conv2 = nn.Conv1d(128, 80, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        # x: [B, 80, T]
        # Gate: calculate energy per frame, smooth step from -14 to -10
        frame_energy = x.mean(dim=1, keepdim=True) # [B, 1, T]
        gate = torch.clamp((frame_energy - (-14.0)) / 4.0, 0.0, 1.0)
        delta = self.conv2(self.act(self.conv1(x)))
        return x + gate * delta

model = CleanTimbreAdaptor()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.SmoothL1Loss()

print("\n[*] Training Clean Timbre Adaptor (Zero-Bias + Energy Preserving)...")
model.train()
for epoch in range(1, 21):
    total_loss = 0.0
    for b_feat, mm_feat in training_samples:
        inp = b_feat.unsqueeze(0)
        tgt = mm_feat.unsqueeze(0)

        optimizer.zero_grad()
        pred = model(inp)
        loss = criterion(pred, tgt) + 0.3 * (1.0 - torch.cosine_similarity(pred, tgt, dim=1).mean())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:02d}/20] Loss: {total_loss / len(training_samples):.4f}")

model.eval()

# 3. Export to ONNX
print("\n[*] Integrating into ONNX model...")
m_base = onnx.load(base_m6_path, load_external_data=False)

for out in m_base.graph.output:
    if out.name == "mel":
        out.name = "mel_adapted"

for node in m_base.graph.node:
    for idx, o in enumerate(node.output):
        if o == "mel":
            node.output[idx] = "mel_pre_adapt"

sd = model.state_dict()
w1 = sd["conv1.weight"].numpy() # [128, 80, 3]
w2 = sd["conv2.weight"].numpy() # [80, 128, 3]

w1_init = numpy_helper.from_array(w1, name="adaptor.clean.conv1.weight")
w2_init = numpy_helper.from_array(w2, name="adaptor.clean.conv2.weight")

# Constants for gating:
# frame_energy = ReduceMean(mel_pre_adapt, axes=[1], keepdims=1)
# gate = Clip((frame_energy - (-14.0)) / 4.0, 0.0, 1.0)
th_init = numpy_helper.from_array(np.array([-14.0], dtype=np.float32), name="adaptor.gate.th")
scale_init = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="adaptor.gate.scale") # 1/4
min_init = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="adaptor.gate.min")
max_init = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="adaptor.gate.max")
m_base.graph.initializer.extend([w1_init, w2_init, th_init, scale_init, min_init, max_init])

# Nodes:
# 1. Delta calculation: Conv1 -> Sigmoid -> Mul -> Conv2
conv1_node = helper.make_node(
    "Conv",
    inputs=["mel_pre_adapt", "adaptor.clean.conv1.weight"],
    outputs=["adaptor.clean.conv1_out"],
    pads=[1, 1],
    name="adaptor.clean.conv1"
)
sig_node = helper.make_node(
    "Sigmoid",
    inputs=["adaptor.clean.conv1_out"],
    outputs=["adaptor.clean.sig_out"],
    name="adaptor.clean.sig"
)
silu_node = helper.make_node(
    "Mul",
    inputs=["adaptor.clean.conv1_out", "adaptor.clean.sig_out"],
    outputs=["adaptor.clean.silu_out"],
    name="adaptor.clean.silu"
)
conv2_node = helper.make_node(
    "Conv",
    inputs=["adaptor.clean.silu_out", "adaptor.clean.conv2.weight"],
    outputs=["adaptor.clean.delta"],
    pads=[1, 1],
    name="adaptor.clean.conv2"
)

# 2. Gate calculation (Opset 13 uses axes attribute)
reduce_node = helper.make_node(
    "ReduceMean",
    inputs=["mel_pre_adapt"],
    outputs=["adaptor.clean.energy"],
    axes=[1],
    keepdims=1,
    name="adaptor.clean.reduce"
)
sub_node = helper.make_node(
    "Sub",
    inputs=["adaptor.clean.energy", "adaptor.gate.th"],
    outputs=["adaptor.clean.sub"],
    name="adaptor.clean.sub"
)
mul_scale_node = helper.make_node(
    "Mul",
    inputs=["adaptor.clean.sub", "adaptor.gate.scale"],
    outputs=["adaptor.clean.scaled"],
    name="adaptor.clean.scale"
)
clip_node = helper.make_node(
    "Clip",
    inputs=["adaptor.clean.scaled", "adaptor.gate.min", "adaptor.gate.max"],
    outputs=["adaptor.clean.gate"],
    name="adaptor.clean.clip"
)

# 3. Gated delta: gate * delta
gate_mul_node = helper.make_node(
    "Mul",
    inputs=["adaptor.clean.delta", "adaptor.clean.gate"],
    outputs=["adaptor.clean.gated_delta"],
    name="adaptor.clean.gate_mul"
)

# 4. Final output: mel_pre_adapt + gated_delta
add_node = helper.make_node(
    "Add",
    inputs=["mel_pre_adapt", "adaptor.clean.gated_delta"],
    outputs=["mel"],
    name="adaptor.clean.residual_add"
)

m_base.graph.node.extend([
    conv1_node, sig_node, silu_node, conv2_node,
    reduce_node, sub_node, mul_scale_node, clip_node,
    gate_mul_node, add_node
])

for out in m_base.graph.output:
    if out.name == "mel_adapted":
        out.name = "mel"

output_onnx_path = "voices/minimax_shaonv/matcha_shaonv_step6.onnx"
onnx.save(m_base, output_onnx_path)
print(f"[+] Successfully exported clean ONNX: {output_onnx_path}")

# 4. Synthesize test cases and inspect audio statistics
test_sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
fleet_bin = os.path.expanduser("~/.rpty/bin/fleet")

test_cases = [
    ("minimax_shaonv_test01_bilingual.wav", "你好，我是 Seeed Studio 智能语音助手。Hello, welcome to the future.", "zh"),
    ("minimax_shaonv_test02_english.wav", "Welcome to Seeed Studio! We make open-source hardware accessible everywhere.", "en"),
]

print("\n" + "=" * 60)
print("=== AUDIO QUALITY & ASR TRANSCRIPTION VALIDATION ===")
print("=" * 60)

for fname, text, lang in test_cases:
    t_ids = vocoder.text_to_tokens(text)
    fixed_ids = []
    for i in range(len(t_ids)):
        fixed_ids.append(t_ids[i])
        if t_ids[i] == 1624 and i + 1 < len(t_ids) and t_ids[i+1] == 48:
            fixed_ids.append(1)

    toks = np.array([fixed_ids], dtype=np.int64)
    xlen = np.array([len(fixed_ids)], dtype=np.int64)

    mel = test_sess.run(["mel"], {
        "x": toks, "x_length": xlen,
        "noise_scale": np.array([0.0], dtype=np.float32),
        "length_scale": np.array([1.0], dtype=np.float32)
    })[0]

    wav = mel2wav(mel)
    peak = np.max(np.abs(wav))
    if peak > 0: wav = wav * (0.90 / peak)

    # Check silence in first 100ms
    head_100ms_rms = np.sqrt(np.mean(wav[:1600]**2))
    tail_100ms_rms = np.sqrt(np.mean(wav[-1600:]**2))
    total_rms = np.sqrt(np.mean(wav**2))
    dur = len(wav) / SAMPLE_RATE

    out_path = f"voices/samples/{fname}"
    sf.write(out_path, wav, SAMPLE_RATE)

    # ASR verification
    with open(out_path, "rb") as f:
        resp = requests.post(f"http://127.0.0.1:8621/asr?language={lang}", files={"file": (fname, f, "audio/wav")})
        asr_txt = resp.json().get("text", "")

    print(f"\n[Case] {fname} ({dur:.2f}s)")
    print(f"  Head Silence RMS: {head_100ms_rms:.6f}")
    print(f"  Tail Silence RMS: {tail_100ms_rms:.6f}")
    print(f"  Total Audio RMS:  {total_rms:.4f}")
    print(f"  Max Peak:         {np.max(np.abs(wav)):.4f}")
    print(f"  Target Text:      {text}")
    print(f"  ASR Result:       {asr_txt}")

    os.system(f"{fleet_bin} push macbook {out_path} /Users/harvest/Downloads/{fname}")
    print(f"  [+] Pushed to MacBook: ~/Downloads/{fname}")
