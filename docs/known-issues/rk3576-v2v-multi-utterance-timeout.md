# RK3576：`--multi 2` 的第二句必然超时，RK 侧 V2V 统计不可用

平台：reComputer RK3576（LubanCat-3 形态，fleet 设备 `cat-remote`），Debian 12 Rockchip BSP。
镜像 `sensecraft-missionpack.seeed.cn/solution/openvoicestream:rk-20260913.3`，
容器 `conversational-voice-speech`，端口 8621。观测日期 2026-09-20。

## 背景：`--multi 2` 是干什么的

`server/main.py` 在 V2V 流里收到第一个 `asr_final` 后立即置 `asr_session_closed=True`
并关闭 socket。单句模式下注入的 `{"type":"text"}` 还没进 TTS 缓冲，会话就断了
（容器日志只有 `v2v stream closed`，没有任何 TTS 活动）。

`bench/perf/measure_v2v_unified.py` 的 `--multi 2`（同一条 wav 间隔 `--silence-ms`（默认 700 ms）重放两次）
就是为绕过这个提前关闭而设计的：靠第二句把会话撑开，让第一句的 TTS 回环跑完。

## 症状

在 RK3576 上第二句**稳定**超时，四轮重复全部一致：

```
每轮第二句           result = null, error = "timeout"
summary 表           全部 "no samples"（脚本只统计完整轮次）
stop_to_tts_audio_ms 1575 / 1624 / 1596 / 1580 ms   （第一句，稳定约 1.6 s）
final_to_tts_audio_ms ~450 ms
stop_to_final_ms     5111 / 5161 / 5128 / 5113 ms   （反常，见下）
```

## 复现

设备空闲（load 0.25，只有 `conversational-voice-speech` 一个容器）：

```bash
python bench/perf/measure_v2v_unified.py \
  --host 127.0.0.1:8621 \
  --wav bench/perf/corpus/short/zh_short_01.wav \
  --tts --multi 2 --runs 5 --warmup 1
```

- 脚本 `bench/perf/measure_v2v_unified.py` md5 `9a3ff631817fb2dc7300d7895db2bed4`
- 音频 `/home/cat/ovs-ps-build-rk/bench/perf/corpus/short/zh_short_01.wav`
  sha256 `e7c9cfc1a6b9d06466c959623fd80e7c19921038f4e6c85af432b1497306cea9`
  （与 Orin NX 上那份逐字节相同）

## 为什么第二句会超时

RK 上 TTS 合成约 450 ms/句（`final_to_tts_audio_ms`）。第一句的 TTS 还没合成完，
服务端已经按 `asr_session_closed` 的逻辑把 socket 关掉，第二句的音频再无处可去，
客户端等到超时。也就是说：`--multi 2` 这个绕法依赖"第二句到达时会话还开着"，
而 RK 上该前提不成立。这是服务端会话生命周期的产品行为问题，不是 bench 脚本的
问题——**不要改 `measure_v2v_unified.py` 去"修"它**，改脚本只会把问题盖住。

## 另一个未解释的反常：`stop_to_final_ms` > `stop_to_tts_audio_ms`

同一轮里 `stop_to_final_ms`（约 5.1 s）**大于** `stop_to_tts_audio_ms`（约 1.6 s）。
文字 final 本应先于合成语音出现，这里顺序是反的，说明两个指标在 RK 上走的不是
同一条通路。成因尚未定位，此处只记录观测，不做推测。

## 结论：RK 侧 V2V 统计不可用

1. 第二句恒 timeout → `summary` 全 "no samples"，脚本给出的 p50/mean 之类聚合值
   在 RK 上没有样本支撑。
2. `stop_to_final_ms` 与 `stop_to_tts_audio_ms` 互相矛盾，**聚合值**（p50/mean）都不能对外引用；唯一可用的是第一句的单次 `stop_to_tts_audio_ms`（见下）。
3. 历史上记入
   `seeed-solutions-hub/.../conversational_voice_ai/model-matrix/rk3576/boundary.zh.yaml`
   的 `stop_to_tts_audio_ms p50=3782ms`（镜像 `rk-20260903.10`）很可能就是这个超时状态下的
   统计产物：本次在 `rk-20260913.3` 上空闲重测，同一指标稳定在约 1.6 s。该 yaml 已按
   本次重测更正并标注 `stop_to_final_ms` 存疑。

唯一目前可用的 RK 侧数字，是**第一句**的 `stop_to_tts_audio_ms` ≈ 1.6 s（n=4，回声模式，
链路中无对话模型）。

> 适用范围：以上现象在镜像 `rk-20260913.3`（voxedge 0.0.15a0）上复现；更新的 RK 镜像未复测，请按当前版本重新验证后再引用本结论。
