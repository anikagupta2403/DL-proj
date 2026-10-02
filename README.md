# Abnormal Crowd Behavior Detection

A deep learning mini project that detects **abnormal behavior in surveillance footage**: classifying scenes as **normal**, **fall** or **fight**. We compare three model families that see the data differently:

| Model | Input | Captures | Status | Branch |
|---|---|---|---|---|
| **2D CNN** | Single frames | Spatial appearance (pose, scene) | ✅ Done | [`2d-cnn`](https://github.com/anikagupta2403/DL-proj/tree/2d-cnn) |
| **3D CNN** | Short clips (stacks of frames) | Spatial + short-range motion | 🔜 Planned | – |
| **Transformer** | Frames or clips as token sequences | Long-range spatial / temporal relations | 🔜 Planned | – |

Each model lives on its own branch with its notebooks, trained weights and a detailed README. `main` holds this overview.

---

## Problem

Surveillance systems produce far more footage than people can watch. Automatically flagging abnormal events, such as people falling or fighting, lets operators respond faster. In this setting, **missing an incident is worse than a false alarm**, so we track per-class recall alongside macro-F1.

## Dataset

[**Abnormal Behavior Detection v1**](https://universe.roboflow.com/abnormal-behavior-detection/abnormal-behavior-detection-5geje/dataset/1) on Roboflow Universe (CC BY 4.0):

- 3,709 surveillance frames at 640×640, extracted from public fall-detection (e.g. UR Fall) and fight video sets
- Originally annotated for **object detection** (YOLO bounding boxes). We convert the boxes to **one label per image**
- 6 raw classes (`0, 1, 2, fall, fight, normal`) merged into 3: **normal / fall / fight**

### Shared preprocessing

All models use the same cleaned data and split, so their results can be compared directly:

```
Raw Roboflow export (YOLOv8)            3,709 images
  ↓ Validation        corrupt images, label format, box ranges
  ↓ Deduplication     109 byte-identical frames removed
  ↓ Class merging     0→normal, 1→fall, 2→fight; image label = fight > fall > normal
  ↓ Group-aware split by source video (no video in more than one split), ≈ 60/20/20
Final dataset                           3,599 images (1,200 fall / 1,199 fight / 1,200 normal)
```

The split is grouped by video because Roboflow's original split interleaves consecutive frames of the same clip, which leaks near-duplicates into validation.

---

## Models

### 1. 2D CNN (done)

Frame-level classification. Two variants were trained on the same split:

| Variant | Params | Test accuracy | Test macro-F1 |
|---|---|---|---|
| ResNet-18 (ImageNet-pretrained, fine-tuned) | 11.18 M | **0.992** | **0.994** |
| Custom 4-block CNN (from scratch) | 1.17 M | 0.969 | 0.969 |

Both models detect every fall and fight in the test set (784 frames); all of their errors are false alarms on normal frames. Architecture, parameters, training curves and the full analysis are in the [`2d-cnn` branch README](https://github.com/anikagupta2403/DL-proj/tree/2d-cnn).

### 2. 3D CNN (planned)

The model convolves over both space and time on short clips of consecutive frames, so it can learn **motion** (collapsing, striking, scattering) instead of single-frame appearance.

### 3. Transformer (planned)

The model uses self-attention over image patches and/or frame sequences to capture long-range relationships across the scene and over time.

---

## Known limitations of the data

- **No crowd panic class.** The dataset covers falls and fights, so the task is framed as general abnormal behavior detection.
- **Scene can give away the class.** Fall frames come mostly from a few indoor rooms and fight frames mostly from outdoor scenes, so a model can rely partly on the background. Near-perfect frame-level scores should be read with this in mind.
- **Limited temporal data.** The dataset contains sampled frames, not full videos. Clip-based models (3D CNN, video transformers) need reconstructed frame sequences or the original source videos.

---

## Repository structure

```
main      ← project overview (this README)
2d-cnn    ← 2D CNN notebooks, trained weights, results, detailed README
3d-cnn    ← (planned)
transformer ← (planned)
```

All training runs on **Kaggle (GPU)**. Each branch's README explains how to reproduce its results.
