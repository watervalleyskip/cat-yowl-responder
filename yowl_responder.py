# -*- coding: utf-8 -*-
"""
猫嚎叫检测自动回应系统 V1（笔记本原型）
检测(YAMNet) -> 播放素材 -> 升级链 -> 冷却闸 -> 安静重置
所有可调参数在 config.json,不要改代码里的数字。

状态机:
  MONITOR        正常检测
  MUTED          播放期间 + 播放后 R 秒,检测结果一律丢弃(防反馈循环)
  FORCED_SILENCE 连播达上限后强制静默 T 分钟,期间只记日志不播放
升级链与"上次触发是否有效"的判定:
  播放后 rearm_window_seconds 内再次触发 => 上一素材记为无效,升级到下一素材
  超过该窗口才再触发 => 上一素材记为有效,但升级位置保持(不自动回退)
  持续安静 quiet_reset_minutes => 升级链重置回素材 A,连播计数清零
"""

import csv
import json
import os
import queue
import sys
import time
import wave
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd

# ---------- 配置加载 ----------

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"), encoding="utf-8") as f:
    CFG = json.load(f)

SR = CFG["audio"]["sample_rate"]                      # 16000, YAMNet 硬性要求
WIN = int(CFG["audio"]["window_seconds"] * SR)        # 0.975s -> 15600 samples
HOP = float(CFG["audio"]["hop_seconds"])
PAUSE_AFTER = float(CFG["detection"]["pause_after_playback_seconds"])
NEED_HITS = int(CFG["detection"]["consecutive_hits"])
REARM = float(CFG["escalation"]["rearm_window_seconds"])
BURST_LIMIT = int(CFG["escalation"]["burst_limit"])
FORCED_SILENCE = float(CFG["escalation"]["forced_silence_minutes"]) * 60
QUIET_RESET = float(CFG["escalation"]["quiet_reset_minutes"]) * 60
MATERIALS = CFG["escalation"]["materials"]
VOLUME = float(CFG["playback"]["volume"])
CLIPS_DIR = CFG["logging"]["clips_dir"]
LOG_CSV = CFG["logging"]["log_csv"]
PRE_S = float(CFG["logging"]["pre_seconds"])
POST_S = float(CFG["logging"]["post_seconds"])

os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

# ---------- 模型加载(TF Hub 自动下载并缓存,首次运行需联网) ----------

print("加载 YAMNet 模型(首次运行会自动下载,约 17MB)...")
import tensorflow as tf  # noqa: E402
import tensorflow_hub as hub  # noqa: E402

model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = model.class_map_path().numpy().decode("utf-8")
with tf.io.gfile.GFile(class_map_path) as f:
    CLASS_NAMES = [row["display_name"] for row in csv.DictReader(f)]

# 从官方 class map 解析目标类别索引,不硬编码猜测
TRIGGER_IDX = {}
for name, thr in CFG["detection"]["trigger_classes"].items():
    if name in CLASS_NAMES:
        TRIGGER_IDX[CLASS_NAMES.index(name)] = (name, float(thr))
    else:
        print(f"[警告] 类别 '{name}' 不在 YAMNet class map 中,已忽略。")
if not TRIGGER_IDX:
    sys.exit("没有任何有效触发类别,检查 config.json 的 trigger_classes。")
print("触发类别:", {v[0]: v[1] for v in TRIGGER_IDX.values()})

# ---------- 播放(pygame,支持 wav/ogg/mp3) ----------

import pygame  # noqa: E402

pygame.mixer.init(frequency=44100)
SOUNDS = []
for p in MATERIALS:
    if not os.path.exists(p):
        sys.exit(f"素材文件不存在: {p}(把素材放进 sounds/ 并在 config.json 里登记)")
    s = pygame.mixer.Sound(p)
    s.set_volume(VOLUME)
    SOUNDS.append(s)
print(f"已加载 {len(SOUNDS)} 个素材,音量 {VOLUME:.0%}")

# ---------- 音频采集:回调只入队,主循环消费 ----------

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

def _callback(indata, frames, t, status):
    if status:
        print("[音频状态]", status, file=sys.stderr)
    audio_q.put(indata[:, 0].copy())

# ---------- 日志 ----------

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp", "class", "confidence", "material", "clip_file", "effective"]
        )

def log_row(ts, cls, conf, material, clip_file, effective=""):
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, cls, f"{conf:.3f}", material, clip_file, effective])

def mark_last_effectiveness(effective: bool):
    """把 CSV 最后一条触发记录的 effective 字段补上(简单实现:整读整写)。"""
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for i in range(len(rows) - 1, 0, -1):
        if rows[i] and rows[i][5] == "":
            rows[i][5] = "yes" if effective else "no"
            break
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

def save_clip(pre_audio: np.ndarray, post_audio: np.ndarray, ts_name: str) -> str:
    path = os.path.join(CLIPS_DIR, f"{ts_name}.wav")
    data = np.concatenate([pre_audio, post_audio])
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path

# ---------- 主循环 ----------

def main():
    ring = deque(maxlen=int((PRE_S + 1.0) * SR))   # 触发前音频的滚动缓冲
    window = np.zeros(WIN, dtype=np.float32)       # 推理窗口
    hits = 0                                       # 连续命中计数
    esc_idx = 0                                    # 升级链位置
    burst = 0                                      # 连播计数
    muted_until = 0.0                              # 播放静默截止时间
    forced_until = 0.0                             # 强制静默截止时间
    last_trigger_t = None                          # 上次触发时刻
    awaiting_effect = False                        # 是否在等待"是否复叫"判定
    pending_post = None                            # (截止时间, 触发时刻名) 等待补存后段音频
    post_buf = []
    last_infer = 0.0

    print("开始监听。Ctrl+C 退出。")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=_callback, device=CFG["audio"]["input_device"]):
        while True:
            chunk = audio_q.get()
            ring.extend(chunk)
            now = time.time()

            # 补存触发后段音频
            if pending_post is not None:
                post_buf.append(chunk)
                if now >= pending_post[0]:
                    post = np.concatenate(post_buf)[: int(POST_S * SR)]
                    pre = np.array(ring, dtype=np.float32)[-int((PRE_S + POST_S) * SR):-len(post)] \
                        if len(post) else np.array(ring, dtype=np.float32)
                    clip = save_clip(pre[-int(PRE_S * SR):], post, pending_post[1])
                    print(f"  片段已存: {clip}")
                    pending_post, post_buf = None, []

            # 升级链有效性判定 + 安静重置
            if awaiting_effect and last_trigger_t and now - last_trigger_t > REARM:
                mark_last_effectiveness(True)
                awaiting_effect = False
                print("  [判定] 上一素材有效(窗口内未复叫)")
            if last_trigger_t and now - last_trigger_t > QUIET_RESET and (esc_idx or burst):
                esc_idx, burst = 0, 0
                print("  [重置] 持续安静,升级链回到素材 A")

            # 静默期:丢弃检测
            if now < muted_until:
                continue

            # 攒够一个 hop 再推理
            if now - last_infer < HOP:
                continue
            last_infer = now
            buf = np.array(ring, dtype=np.float32)
            if len(buf) < WIN:
                continue
            window = buf[-WIN:]

            scores = model(window)[0].numpy().max(axis=0)  # 各帧取最大
            fired = None
            for idx, (name, thr) in TRIGGER_IDX.items():
                if scores[idx] >= thr:
                    fired = (name, float(scores[idx]))
                    break
            if fired is None:
                hits = 0
                continue
            hits += 1
            print(f"命中 {fired[0]} conf={fired[1]:.2f} ({hits}/{NEED_HITS})")
            if hits < NEED_HITS:
                continue
            hits = 0

            # ---- 正式触发 ----
            ts = datetime.now()
            ts_name = ts.strftime("%Y%m%d_%H%M%S")

            # 强制静默期:只记日志不播放
            if now < forced_until:
                print(f"[强制静默中] 记录但不播放,剩余 {forced_until - now:.0f}s")
                log_row(ts.isoformat(timespec="seconds"), fired[0], fired[1],
                        "(forced_silence)", "", "n/a")
                last_trigger_t = now
                continue

            # 复叫 => 上一素材无效,升级
            if awaiting_effect:
                mark_last_effectiveness(False)
                esc_idx = min(esc_idx + 1, len(SOUNDS) - 1)
                print(f"  [判定] 复叫,升级到素材 {esc_idx}")

            material = MATERIALS[esc_idx]
            print(f"==> 触发!播放 {material}(第 {burst + 1}/{BURST_LIMIT} 连播)")
            ch = SOUNDS[esc_idx].play()
            dur = SOUNDS[esc_idx].get_length()
            muted_until = now + dur + PAUSE_AFTER   # 播放期 + R 秒停检
            log_row(ts.isoformat(timespec="seconds"), fired[0], fired[1], material, f"{ts_name}.wav")
            pending_post = (now + dur + POST_S, ts_name)
            post_buf = []
            last_trigger_t = now
            awaiting_effect = True
            burst += 1
            if burst >= BURST_LIMIT:
                forced_until = now + FORCED_SILENCE
                burst = 0
                print(f"[闸] 连播达上限,强制静默 {FORCED_SILENCE / 60:.0f} 分钟")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止。日志:", LOG_CSV)
