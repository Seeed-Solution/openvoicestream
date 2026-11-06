# Spec — 通用说话人分离能力(Diarization)

> 承接 `ovs-punct-speaker-handoff.md`。那份 handoff 把「OVS 只吐声纹向量、辨认/匹配留给使用端」锁死,并 ship 了**按句(每个 VAD 段)产出 CAM++ embedding** 的能力。本 spec 补的是当时明确 defer 的那块:**说话人分离本体**(聚类盲分 + 可选辨认),支持流式 / 非流式,跨所有 ASR backend 通用。
>
> **一句话结论**:边缘层(各 ASR backend)只多做一件事——给每段已产出的向量补 `start/end` 时间戳;**分离逻辑是纯 numpy 聚类、设备无关、零额外模型**,落在服务层;辨认(出人名)走可选 registry、默认关、是消费端责任。

---

## 0. 为什么这样切

三条既有事实决定了边界,不能违背:

1. **ASR backend 接口不带段级时间戳**(`voxedge/backends/base.py` 的 `TranscriptionResult` 只有 text/language/meta)→ 不改 backend,段边界从 **silero VAD 的 speech_start/end 事件**拿。
2. **「OVS 只吐向量,匹配是使用端的事」已锁定**(handoff §2.3)→ 聚类(盲分)可在服务层做,但**出人名的 registry 匹配默认不做、留给消费端**。
3. **每段一个向量已经有了**(handoff §3,流式 `?speaker_embedding=true` 在 finalize 对该 VAD 段跑一次 CAM++)→ diarization 不需要新模型、不需要重切音频,**只在已有向量序列上做聚类**。

因此分离 = **已有的「按段向量」 + 新增的「numpy 聚类」**。聚类不依赖任何引擎,所以**所有设备(Jetson/RK/RPi/CPU)天然通吃**;唯一设备相关的是「出向量」那层(CAM++ 跑在哪),那部分复用已有 backend 选择机制,并单独排 TRT engine 化。

---

## 1. 分层架构

```
┌─ 边缘 / 能力层 (各 ASR backend, 设备相关) ──────────────────┐
│  ASR backend  → text                                        │
│  CAM++ 提向量 → 每个 VAD 段一个 192-d 向量  (已 ship)        │
│       + start/end 时间戳                    (本 spec 小增量) │
│  【只吐向量,绝不聚类、不出人名】                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ (start, end, embedding)  序列
┌─ 服务层 (纯 numpy, 设备无关) ─────────────────────────────────┐
│  盲分 Diarization:                                           │
│    online  → 增量聚类,每段即时贴 spk_N      (流式)          │
│    offline → 整段重聚类,估计说话人数         (非流式)       │
│  【新增,本 spec 主体。零模型,只有 numpy】                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ (可选, 默认 OFF)
┌─ 辨认 Identification (消费端责任) ────────────────────────────┐
│  spk_N 簇中心 → 余弦比对声纹注册库 → 张三/李四                │
│  【registry + 阈值 + 身份,本 spec 只给 opt-in 接口,默认不做】│
└──────────────────────────────────────────────────────────────┘
```

代码落点(复刻 `speaker_embedding` 的 voxedge-kernel + product-shim 双层):

| 层 | 文件 | 职责 |
|---|---|---|
| 聚类内核(env-free) | `voxedge/voxedge/capabilities/diarization.py` | 纯 numpy:`OnlineDiarizer` / `OfflineDiarizer` / `SpeakerSegment` |
| 产品编排 + 端点 | `seeed-local-voice/server/core/diarization.py` | session 状态、feature flag、懒加载、端点 handler |
| 边缘增量 | `seeed-local-voice/server/main.py`(finalize 点 :2454/:2510 一带) | 给每段 embedding payload 补 `start/end` |

---

## 2. 数据结构

```python
@dataclass
class SpeakerSegment:
    start: float            # 秒,相对 session 起点
    end: float
    speaker: str            # 匿名 "spk_0" / "spk_1" …;辨认开启后可被人名覆盖
    confidence: float       # 与所属簇中心的 cosine
    embedding: Optional[np.ndarray] = None   # 192-d,可选回传(默认不带,省带宽)
```

向量编码沿用已有契约(`voxedge.capabilities.speaker_embedding.encode_embedding`,little-endian f4 base64),`embedding_model` / `dim` / `normalized` 元数据照带——换模型时旧聚类/注册全失效,靠版本号检测。

---

## 3. 聚类内核(voxedge, 纯 numpy)

### 3.1 在线(流式)— `OnlineDiarizer`

维护「簇中心列表」,每来一段即时归类,延迟 ≈ 一个 VAD 段。

```python
class OnlineDiarizer:
    def __init__(self, threshold=0.55, ema=0.7, max_speakers=10): ...
    def assign(self, emb, start, end) -> SpeakerSegment:
        # 1. 与每个现有中心算 cosine,取 max
        # 2. max >= threshold → 归入该簇;中心 = ema*中心 + (1-ema)*emb,再 L2 归一
        # 3. 否则且 簇数 < max_speakers → 新建 spk_{k}
        # 4. 满了 → 归到最近簇(不再增)
    def relabel(self) -> list[SpeakerSegment]:
        # session 结束可选:用累积向量跑一次 offline 聚类,
        # 把在线临时标签映射到稳定标签(纠正"先到先得割裂同一人")
```

- 说话人数**未知、动态增长**。
- `threshold` 是核心调参(同源 CAM++ 同人 cosine 常 >0.6,异人 <0.4;0.55 是稳妥默认,分设备/场景可在 leaf 覆盖)。
- `relabel()` 解决在线聚类固有缺陷:同一人若中途被误判成新簇,会后重聚类能合回。

### 3.2 离线(非流式)— `OfflineDiarizer`

收齐全部段向量后整体聚类,质量最优,自动估计说话人数。

```python
class OfflineDiarizer:
    def cluster(self, segs: list[(emb,start,end)],
                num_speakers: Optional[int]=None) -> list[SpeakerSegment]:
        # 相似度矩阵 = cosine(归一化向量两两点积)
        # num_speakers 给定 → 凝聚层次聚类(AHC, average linkage)切 k 簇
        # 未给定 → 阈值停止的 AHC,或谱聚类 + eigengap 估 k
```

- **聚类是纯线性代数**(相似度矩阵 + AHC/谱聚类),numpy/scipy 即可,**不引入任何模型、不依赖 sherpa**。`OfflineSpeakerDiarization` 那种打包件不用(它捆 pyannote 分段,我们已有 VAD)。
- 估计说话人数:相似度矩阵特征值 eigengap,或 AHC 在 `1 - threshold` 距离处切。提供 `num_speakers` 可选 hint 走更稳的固定 k 路径。

### 3.3 为什么内核放 voxedge 且 env-free

与 `speaker_embedding.py` 同构:纯算法、无 env、无 IO,任何镜像可用;product 层注入参数(threshold 等来自 leaf)、管 session 生命周期。

---

## 4. API 契约

### 4.1 流式(在线盲分)

`/asr/stream`、`/v2v/stream` 加开关 `?diarize=true`(env 默认 `OVS_DIARIZE`,query 覆盖,照 `?vad=` / `?speaker_embedding=` 惯例)。

- 开启 `diarize` **隐含开启 speaker_embedding**(聚类要向量;但向量默认不回传,除非也显式要 embedding)。
- 每个 final 增加字段:

```jsonc
// ?diarize=true
{ "type":"final", "text":"你好,世界。",
  "speaker":"spk_0", "speaker_conf":0.83,
  "start":1.20, "end":2.65 }              // start/end 相对 session 起点
```

- session 结束(断连 / 显式 EOS)可选回一条 `{type:"diarization_summary", segments:[...], num_speakers:2}`,内部跑 `relabel()` 给出纠偏后的全局标签。

### 4.2 非流式(离线盲分)

```
POST /diarize
  body: audio (WAV/PCM16 16k mono, 可多说话人长音频)
  query: ?num_speakers=N (可选), ?return_embeddings=true (可选)
  resp: { "num_speakers":2,
          "segments":[ {"start":0.0,"end":3.2,"speaker":"spk_0","confidence":0.88}, ... ],
          "embedding_model":"campplus_sv_zh_en_3dspeaker", "dim":192 }
```

内部:VAD 切段 → 每段 CAM++ 向量 → `OfflineDiarizer.cluster()`。复用现有 `/speaker/embedding` 的解码/重采样/提向量路径。

### 4.3 辨认(可选,默认 OFF,消费端责任)

`?identify=true`(流式)或 `POST /diarize?identify=true`:对每个簇中心比对一个声纹注册库,匹配上则把 `speaker` 从 `spk_N` 覆盖成已知人名。

- 默认关闭——**这是 handoff §6 锁定的「使用端责任」**。OVS 这里只提供 opt-in 钩子,registry 可注入(复用 `tts_speakers.py` 那种「只存向量、JSON 持久化」的轻量库,或外部回调)。
- 未匹配的人仍保留匿名 `spk_N`(开会来个没注册的人不丢)。
- 阈值、未注册判定、库管理全部可配 / 可外置。

---

## 5. 边缘增量(本 spec 唯一改 main.py 的地方)

现状:`?speaker_embedding=true` 已在每段 finalize 跑 CAM++ 出向量。**只补时间戳**:

- session 起点记一个 `t0`(首个 chunk 时间);累计已喂入样本数 / sample_rate → 当前时间轴。
- finalize 时该 VAD 段的 `[start,end]` = 段缓冲首样本时刻 → 末样本时刻(段缓冲已存在,handoff §4「一句一攒、finalize 一清」)。
- 把 `start/end` 加进 final payload。**纯加字段,不破坏现有契约**;不开 diarize/embedding 时零开销零行为变化。

时间戳是**段级**(VAD 端点推),不是 word 级——足够 diarization 排时间轴。

---

## 6. 分设备矩阵

| 设备 | 出向量(CAM++) | 聚类 | 流式 | 非流式 |
|---|---|---|---|---|
| Jetson Orin NX/Nano | **TRT engine**(分轨,见 §7) | numpy | ✅ parity 0.99999 | ✅ |
| RK3576/3588 | CPU sherpa-onnx(NPU 让给 ASR) | numpy | ✅ 实测低并发(≤~3 流) | ✅ |
| RPi 5(A76) | CPU sherpa-onnx | numpy | ✅ 实测(≤~6 流) | ✅ |
| RPi 4(A72) | CPU sherpa-onnx | numpy | ✅ 实测低并发(≤~3 流) | ✅ |

- **聚类对所有设备一致**(numpy,~µs 级,与设备无关)。瓶颈只在每段一次 CAM++ 前向。
- **RK3576 实测**(reComputer RK3576,2026-06-26):CAM++ sherpa CPU RTF≈0.09–0.13(1s→125ms/3s→255ms/5s→428ms,2 线程),聚类 N=10 仅 1.45ms。每句一次 embedding 在 VAD finalize 后算、不阻塞音频流 → **online 低并发(≤~3 流)实时可行**,offline 无条件可行;NPU 留给 ASR,不需要 TRT。多并发 finalize 时 CPU 竞争近线性(~RTF 0.1×N)。
- **RPi 5 实测**(harvest-pi,A76 4 核,2026-06-26):比 RK3576 快 ~3.7×。RTF≈0.03(1s→34ms/3s→90ms/5s→158ms),冷加载 0.6s,聚类 N=10 仅 0.45ms。**online 实时可行、并发上限 ~6 流**,offline 无条件;CAM++ 走 CPU,不需要 Hailo NPU。
- **RPi 4 实测**(seeed-pi,A72 4 核,2026-06-26):RTF≈0.10(1s→114ms/3s→303ms/5s→508ms),冷加载 1.66s。**online 低并发(≤~3 流)可行**,offline 无条件 —— 与 RK3576 同档,约为 Pi 5 的 1/3.5 速度。三方对照:Pi5(A76)≫ RK3576 ≈ Pi4(A72)。盲分标签三机字节一致(同模型)。
- 通过 `ConcurrencyCapability` 声明 diarization 的 `vram_mb_per_slot` / `max_concurrent`,纳入 `capability_resolver` 的 session ceiling,与 ASR/TTS 错峰共享资源。

---

## 7. Jetson TRT engine 出向量(分轨, 独立技术风险)

跟 ASR/TTS 一样,Jetson 上 CAM++ 走**常驻 TRT engine**,不挂 onnxruntime。做成 `EmbeddingExtractor` 抽象的一个后端,**聚类层完全不变**:

```python
class EmbeddingExtractor(ABC):
    def extract(self, audio, sr) -> np.ndarray: ...   # → 192-d L2-norm
# 实现:jetson.campplus_trt(TRT) / cpu.sherpa_campplus(现有, 兜底)
```

要点:
1. **特征前端不进 engine**:engine 吃 `[1,T,80]` fbank,出 `[1,192]`;fbank 在 Python 侧算(kaldi-native-fbank),可复用 ASR 已算 fbank。
2. **变长 = optimization profile**:min/opt/max 时间帧三档;statistics pooling 天然支持变长。
3. **算子兼容 spike(先做)**:stats pooling 的 mean/std、Res2Net split/concat 在 TRT 上可能要手术(参考 BiCodec `ScatterND→Concat` 经验)。先做 **ONNX→engine + 数值对齐**(TRT vs onnxruntime,固定音频 cosine > 0.9999),fp16 验不溢出,再排期。
4. **engine 预构建入 artifacts**,设备端不现场 build;每算力档(orin-nx/orin-nano)各一个 `.plan`。env 用 `DIAR_CAMPPLUS_ENGINE_FILE` 指文件不指目录(踩过的 `*_ENGINE_FILE` 坑),缺 `.meta.json` 时 `_write_meta` 生成。

---

## 8. Leaf 配置

```yaml
# configs/leaves/diarization.yaml
leaves:
  diar.campplus.cpu:                    # RPi/RK/Mac 兜底
    capability: diarization
    backend: cpu.sherpa_campplus
    mode: both                          # online | offline | both
    artifacts: { files: [campplus_sv_zh_en.onnx] }
    params: { online_threshold: 0.55, offline_min_sim: 0.50, max_speakers: 10, min_segment_ms: 600 }
    resources: { peak_unified_mb: 120 }
  diar.campplus.orin-nx:                # Jetson,TRT engine 出向量
    capability: diarization
    backend: jetson.campplus_trt
    device: orin-nx
    artifacts: { files: [engines/campplus_fp16_orin-nx.plan] }
    runtime_env: { DIAR_CAMPPLUS_ENGINE_FILE: /opt/models/diar/campplus_fp16.plan }
    params: { online_threshold: 0.55, offline_min_sim: 0.50, max_speakers: 10, min_segment_ms: 600 }
    resources: { peak_unified_mb: 90 }
```

opt-in 默认 OFF:不挂 diarization leaf 且 `OVS_DIARIZE` 未开 → 完全 no-op,零开销零行为变化(与 speaker_embedding 一致)。

---

## 9. 分阶段落地

| 阶段 | 内容 | 依赖设备 | 验收 |
|---|---|---|---|
| **P0a** | 聚类内核(voxedge `diarization.py`:Online/Offline + SpeakerSegment)+ 单测(合成向量) | 无(CPU) | 已知簇结构的合成向量,聚类标签正确率 |
| **P0b** | 边缘:每段 embedding 补 start/end | 无 | final payload 含正确 start/end |
| **P1** | 端点接线:`POST /diarize` + 流式 `?diarize=true` + product 编排/懒加载/flag | 无(CPU sherpa 提向量) | 真双人音频离线分离 ≥ 期望簇数;流式逐段贴标签 |
| **P2** | `relabel()` 会后纠偏 + `diarization_summary` 事件 | 无 | 在线误分的同人会后合回 |
| **P3** | 可选辨认 `?identify=true` + 轻量 registry 注入(默认 off) | 无 | 注册人匹配出名、未注册兜底 spk_N |
| **P4** | Jetson CAM++ TRT engine(§7,先 spike) | Jetson | TRT vs ORT cosine>0.9999;端到端流式分离 |
| **P5** | RK/RPi 性能实测,按预算定 online/offline 开关,写进 profile | RK/RPi | 各设备实测延迟/并发达标 |

P0–P3 全部 CPU 可验证、不需设备,是真正「新做」的主体;P4/P5 是设备适配,与聚类逻辑解耦、可并行。

---

## 10. 验证(voice-agent-e2e)

- 合成多说话人音频(拼接不同人 WAV 段)→ 离线 `/diarize` 校验簇数 + 段边界。
- scripted audio 流式注入 → WS probe 校验每个 final 的 `speaker/start/end` 字段、`?diarize` on/off 行为差异。
- 边界:多句连续、reset、断连、barge-in(v2v)下 session 聚类状态正确重置 / 清空。
- 关闭时回归:不开任何 flag,现有 `/asr/stream` final 字节级不变。
</content>
</invoke>
