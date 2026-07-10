# cat-yowl-responder

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

An automated overnight cat-yowl intervention system powered by YAMNet.

The program continuously listens for cat vocalizations, automatically plays escalating deterrent sounds when yowling is detected, and records every event for later behavioral analysis. It is designed as a behavioral experiment rather than a guaranteed solution — it helps determine whether acoustic intervention is effective for an individual cat while collecting evidence about *why* the cat is vocalizing (heat cycle, food-seeking, attention-seeking).

## Architecture

```
      Microphone
          │
          ▼
 Sliding Audio Window (0.975 s)
          │
          ▼
        YAMNet
   (521 AudioSet classes)
          │
          ▼
  Target Class Detected?
          │
     ┌────┴────┐
     │         │
    No        Yes
     │         │
     ▼         ▼
  Continue   Play Response Clip
                 │
                 ▼
        Cooldown / Mute Logic
                 │
                 ▼
       CSV Log + Audio Snapshot
```

## Features

### Real-time cat vocalization detection
Uses Google's AudioSet-pretrained YAMNet model to classify continuous microphone input without requiring a custom training dataset.

### Escalating response strategy
User-provided sounds are played in an A → B → C escalation chain, allowing progressively stronger responses if yowling continues.

### Echo-safe playback
Detection is automatically suspended while playback is active and for several seconds afterward, preventing the system from triggering on its own output. The **Hiss** class is also excluded as a trigger, since it matches the playback material itself.

### Automatic cooldown
After a configurable number of consecutive responses, playback is force-muted for several minutes to avoid excessive stimulation; after a sustained quiet period, the escalation chain resets.

### Event logging
Every trigger is written to CSV with timestamp, class, confidence score, selected response clip, and whether the vocalization stopped afterward.

### Audio retention
Several seconds of audio surrounding each trigger are preserved for manual inspection of false positives and missed detections.

### Offline operation
Runs entirely locally after the initial YAMNet download.

## How it works

- The microphone continuously captures 16 kHz mono audio.
- Audio is processed using overlapping 0.975-second windows.
- Each window is classified by YAMNet (AudioSet-pretrained, 521 classes).
- A trigger fires only when a target class (**Caterwaul** / **Meow** / **Cat**) exceeds its confidence threshold for N consecutive windows.
- The corresponding deterrent clip is played along the escalation chain.
- Playback enters cooldown before monitoring resumes.
- Every event is logged for later analysis (`summarize.py`).

## Why YAMNet?

YAMNet is a lightweight audio classifier pretrained on Google's AudioSet dataset, covering 521 environmental sound classes. Although it was not trained specifically for pet behavior analysis, it already recognizes **Caterwaul**, **Meow**, and **Cat** as distinct classes — making it a practical zero-shot detector without collecting and labeling a custom dataset.

## Requirements and installation

- Windows
- Python 3.9+
- CPU inference (no GPU required)

```
pip install sounddevice numpy tensorflow tensorflow_hub pygame
```

The first run downloads the YAMNet model (~17 MB); all subsequent runs work completely offline.

## Usage

### 1. Prepare response clips

Collect CC0-licensed cat hiss/growl recordings (e.g. freesound.org, Pixabay), 2–5 seconds each, with no background voices or music. Place them in `sounds/` and list them in `config.json` under `materials`, ordered from weakest to strongest.

### 2. Prevent the computer from sleeping (once)

```
powercfg /change standby-timeout-ac 0
```

### 3. Start monitoring

```
python yowl_responder.py
```

### 4. Review the previous night

```
python summarize.py [YYYY-MM-DD]
```

Then spot-check the saved audio snapshots in `clips/`.

## Configuration

Nearly every runtime parameter lives in `config.json`:

```json
{
  "thresholds": {
    "Caterwaul": 0.25,
    "Meow": 0.40,
    "Cat": 0.50
  },
  "consecutive_hits": 2,
  "burst_limit": 4,
  "forced_silence_minutes": 5,
  "volume": 0.9,
  "materials": ["hiss_A.wav", "hiss_B.wav", "hiss_C.wav"]
}
```

## Example log

```
timestamp,class,confidence,clip,stopped
02:15:37,Caterwaul,0.31,hiss_B.wav,True
03:42:18,Meow,0.56,hiss_C.wav,False
```

## Performance

> Measured values to be added after benchmarking. Planned methodology: sample process CPU% and RSS memory via `psutil` at 1 s intervals over a 30-minute idle-listening session; measure per-window inference latency as the wall-clock time of a single YAMNet forward pass, averaged over 1,000 windows.

| Metric | Value | Environment |
|---|---|---|
| Idle CPU usage (listening) | TBD | TBD (CPU model, Windows version) |
| Per-window inference latency | TBD | TBD |
| Memory footprint (RSS) | TBD | TBD |
| Cold start (model load) | TBD | TBD |

## Tuning

| Symptom | Adjustment |
|---|---|
| Too many false positives | Raise the class threshold; drop the **Cat** class entirely; increase `consecutive_hits` 2 → 3 |
| Missed detections | Lower **Caterwaul** / **Meow** thresholds; check mic-to-cat distance |
| No reaction from the cat | Raise `volume` night by night; swap in different clips |
| Disturbing human sleep | Lower `burst_limit`; raise `forced_silence_minutes` |

Recommended default thresholds:

| Class | Threshold |
|---|---|
| Caterwaul | 0.25 |
| Meow | 0.40 |
| Cat | 0.50 |

**Caterwaul** is the primary target, but the model's confidence for it is systematically low, hence the lowest threshold. **Cat** is the broadest and most false-positive-prone class, hence the highest.

## Limitations

- Detection accuracy depends on the individual cat, microphone placement, and room acoustics. Thresholds must be tuned night by night from the retained trigger audio; there is no out-of-the-box guarantee.
- Acoustic intervention addresses the symptom, not the cause. If triggers cluster at a fixed early-morning hour, the cat is most likely asking for food — a timed feeder targets the cause directly.

## Roadmap

- Raspberry Pi deployment
- ONNX Runtime inference
- Adaptive thresholds
- Multi-cat support
- Web dashboard

## License

MIT
