# Abnormal Crowd Behavior Detection

A deep learning mini project that detects **abnormal behavior in surveillance footage**: classifying scenes as **normal**, **fall** or **fight**. We compare three model families (2D CNN, 3D CNN and Transformer) that see the data differently, with four trained models:

| Model | Family | Input | Captures | Branch |
|---|---|---|---|---|
| **ResNet-18** (ImageNet-pretrained) | 2D CNN | Single frames | Spatial appearance (pose, scene) | [`resnet`](https://github.com/anikagupta2403/DL-proj/tree/resnet) |
| **Custom CNN** (from scratch) | 2D CNN | Single frames | Spatial appearance (pose, scene) | [`2d-cnn`](https://github.com/anikagupta2403/DL-proj/tree/2d-cnn) |
| **Crowd3DCNN** (from scratch) | 3D CNN | 4-frame clips | Spatial + short-range motion | [`3d-cnn`](https://github.com/anikagupta2403/DL-proj/tree/3d-cnn) |
| **ViT-B/16** (ImageNet-pretrained) | Transformer | Single frames as patch tokens | Global spatial relations (self-attention) | [`transformer`](https://github.com/anikagupta2403/DL-proj/tree/transformer) |

Each model lives on its own branch with its executed notebook, trained weights (except the 343 MB ViT checkpoint, which is over GitHub's 100 MB limit), result figures and a detailed README. `main` holds this overview.

---

## Problem

Surveillance systems produce far more footage than people can watch. Automatically flagging abnormal events, such as people falling or fighting, lets operators respond faster. In this setting, **missing an incident is worse than a false alarm**, so we track per-class recall alongside macro-F1.

## Dataset

[**Abnormal Behavior Detection v1**](https://universe.roboflow.com/abnormal-behavior-detection/abnormal-behavior-detection-5geje/dataset/1) on Roboflow Universe (CC BY 4.0):

- 3,709 surveillance frames at 640×640, extracted from public fall-detection (e.g. UR Fall) and fight video sets
- Originally annotated for **object detection** (YOLO bounding boxes). We convert the boxes to **one label per image**
- 6 raw classes (`0, 1, 2, fall, fight, normal`) merged into 3: **normal / fall / fight**

### Shared preprocessing

All models use the same cleaned data and the same split by source video:

```
Raw Roboflow export (YOLOv8)            3,709 images
  ↓ Validation        corrupt images, label format, box ranges
  ↓ Deduplication     109 byte-identical frames removed
  ↓ Class merging     0→normal, 1→fall, 2→fight; image label = fight > fall > normal
  ↓ Group-aware split by source video (no video in more than one split), ≈ 60/20/20
Final dataset                           3,599 images (1,200 fall / 1,199 fight / 1,200 normal)
```

The split is grouped by video because Roboflow's original split interleaves consecutive frames of the same clip, which leaks near-duplicates into validation. The 3D CNN rebuilds this exact split (and checks the frame counts) before building clips, so it is tested on the same held-out videos.

---

## Models

### 1. 2D CNN: ResNet-18 ([`resnet`](https://github.com/anikagupta2403/DL-proj/tree/resnet))

An ImageNet-pretrained ResNet-18 with its head replaced by `Dropout(0.3) → Linear(512, 3)`, fine-tuned end to end for 15 epochs. **Best model overall:** 6 errors out of 784 test frames, all normal frames predicted as fall. It detects every fall and fight.

### 2. 2D CNN: Custom CNN ([`2d-cnn`](https://github.com/anikagupta2403/DL-proj/tree/2d-cnn))

A small VGG-style network (4 conv blocks, 32 → 256 channels, global average pooling) trained from scratch for 40 epochs, as a no-pretraining baseline. It reaches 0.969 macro-F1 with 10× fewer parameters than ResNet-18. Its 24 errors are all false alarms on normal frames.

### 3. 3D CNN ([`3d-cnn`](https://github.com/anikagupta2403/DL-proj/tree/3d-cnn))

A small 3D CNN (3 Conv3d blocks, ~71k parameters) trained from scratch on clips of 4 consecutive frames, so it can learn **motion** and not only appearance. Clips are rebuilt from the Roboflow filenames (source video + frame number). Training clips overlap (stride 1, 884 clips) to get enough data; validation and test clips don't (stride 4), so each evaluation frame is counted once.

It is the weakest model (0.873 macro-F1 on 84 test clips). It catches every fall and fight, but all 10 errors are normal clips from fall videos (the person before or after the fall), so it mostly recognises the fall rooms rather than the motion. The frames are sparse (up to 20 frame numbers apart), so there is little real motion to learn, and validation has only 5 normal clips, which made model selection noisy.

### 4. Transformer: ViT-B/16 ([`transformer`](https://github.com/anikagupta2403/DL-proj/tree/transformer))

An ImageNet-pretrained **ViT-B/16** fine-tuned for 15 epochs. Each frame is cut into 196 patches of 16×16 pixels, and 12 self-attention layers relate every patch to every other. It comes close to ResNet-18 but does not beat it, despite almost 8× more parameters. It has the fewest false alarms (3), but it is the only model that misses falls (9 falls predicted as normal).

### Results

| Model | Params | Pretrained | Test accuracy | Test macro-F1 | Fall recall | Normal recall |
|---|---|---|---|---|---|---|
| ResNet-18 | 11.18 M | ImageNet | **0.992** | **0.994** | 1.000 | 0.983 |
| ViT-B/16 | 85.80 M | ImageNet | 0.985 | 0.987 | 0.969 | **0.992** |
| Custom CNN | 1.17 M | No | 0.969 | 0.969 | 1.000 | 0.932 |
| 3D CNN | ~0.07 M | No | 0.881 | 0.873 | 1.000 | 0.643 |

Fight recall is 1.000 for all four models.

- The three 2D models are scored on the same 784 test frames, so differences of a few frames are within noise.
- The 3D CNN is scored on 84 non-overlapping clips (336 frames) from the same test videos. Only videos whose filenames give a video name and frame number can be turned into clips, so its numbers show the trend but are not an exact head-to-head.
- **Pretraining matters more than model size or architecture:** both ImageNet-pretrained models are clearly ahead of both from-scratch models.
- **Missed incidents are rare:** only the ViT misses abnormal events; all other errors are false alarms on normal frames.

---

## Known limitations of the data

- **Scene can give away the class.** Fall frames come mostly from a few indoor rooms and fight frames mostly from outdoor scenes, so a model can rely partly on the background. Near-perfect frame-level scores should be read with this in mind.
- **Limited temporal data.** The dataset contains sampled frames, not full videos. Clip-based models need reconstructed frame sequences: the 3D CNN gets only short, irregular 4-frame clips, and 1,640 frames without a video name in their filename can't be used at all.

---

## Repository structure

```
main        ← project overview (this README)
resnet      ← ResNet-18 notebook, trained weights, results, detailed README
2d-cnn      ← Custom CNN notebook, trained weights, results, detailed README
3d-cnn      ← 3D CNN notebook, trained weights, results, detailed README
transformer ← ViT-B/16 notebook, results, detailed README (weights too large for GitHub)
```

All training runs on **Kaggle (GPU)**. Each branch's README explains how to reproduce its results.
