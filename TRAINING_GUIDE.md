# YOLO Custom Training Guide
## Teaching the robot to see YOUR farm obstacles

---

## Why you currently only see "Person"

The system ships with **YOLOv8n pretrained on the COCO dataset** — a general-purpose
dataset with 80 object categories. Of those 80, our filter keeps only the ones relevant
to farm safety:

| Detected now (COCO) | NOT detected (need training) |
|---------------------|------------------------------|
| Person ✅           | Rock ❌                       |
| Bicycle ✅          | Tractor ❌                    |
| Car ✅              | Tree / branch ❌              |
| Motorcycle ✅       | Irrigation pipe ❌            |
| Bus / Truck ✅      | Debris / trash ❌             |
| Dog / Cat ✅        | Bales / crates ❌             |
| Horse / Cow ✅      | Anything farm-specific ❌     |

**To detect farm-specific obstacles you must train a custom model on your own images.**
This guide walks you through the complete 5-step process.

---

## Prerequisites

```powershell
pip install ultralytics roboflow   # already in requirements.txt
```

You also need a **free Roboflow account** at [roboflow.com](https://roboflow.com)
(free tier allows unlimited public datasets).

---

## Step 1 — Collect Images from Your Farm Camera

Run the data collector tool. It shows a live camera feed and saves frames into
per-class sub-folders ready for upload.

```powershell
python tools/collect_training_data.py
```

**In the window:**

| Key | Action |
|-----|--------|
| `SPACE` | Save current frame |
| `A` | Toggle auto-capture every 2 seconds |
| `1`–`7` | Switch active class (shown on screen) |
| `Q` | Quit |

**Classes shown on screen:**

| Key | Class |
|-----|-------|
| 1 | person |
| 2 | rock |
| 3 | tractor |
| 4 | tree |
| 5 | irrigation_pipe |
| 6 | animal |
| 7 | debris |

**How to collect good data:**

For each obstacle type, capture images from:
- **3 distances** — close (1m), mid (3m), far (8m+)
- **3 angles** — straight-on, 45° left, 45° right
- **2 lighting conditions** — bright sun, overcast / shade
- **Minimum 50 images per class** (100+ = better accuracy)
- Also capture **10–20 empty field images** (no obstacles) — this reduces false positives

Images are saved to:
```
training/dataset/images/raw/
    rock/
        rock_1715000000000.jpg
        rock_1715000001234.jpg
    tractor/
        tractor_...jpg
    person/
        ...
```

---

## Step 2 — Label Images with Roboflow

Roboflow is a free online tool for drawing bounding boxes around objects.

1. Go to **[roboflow.com](https://roboflow.com)** → Sign up (free)
2. Click **New Project** → choose **Object Detection**
3. Name it `farm-obstacles` → Create
4. Click **Upload** → drag the entire `training/dataset/images/raw/` folder
5. Wait for upload to complete

**Draw bounding boxes:**
- Click an image → draw a box around the obstacle → assign a class label
- Use exactly these labels (spelling matters):
  - `person`, `rock`, `tractor`, `tree`, `irrigation_pipe`, `animal`, `debris`
- Press `W` to switch to next image quickly

**Speed tip — use Auto-Label:**
- After labeling ~20 images manually, click **Auto-Label**
- Roboflow trains a quick model on your labels and pre-labels the rest
- You just correct mistakes instead of drawing every box from scratch

**Once all images are labeled:**
1. Click **Generate** (top menu)
2. Set split: **Train 80% / Val 15% / Test 5%**
3. Add augmentations: ✅ Flip Horizontal, ✅ Brightness ±15%, ✅ Crop 0–20%
4. Click **Generate** → then **Export Dataset**
5. Choose format: **YOLOv8** → **Download zip**

---

## Step 3 — Prepare the Dataset

Extract the downloaded zip into `training/dataset/`:

```
training/dataset/
    images/
        train/      ← ~80% of your images (.jpg)
        val/        ← ~15% of your images (.jpg)
        test/       ← remaining (optional)
    labels/
        train/      ← one .txt per image (same filename, different extension)
        val/
    data.yaml       ← Roboflow generates this — you can ignore it
```

> **Tip:** Roboflow's export zip already has this exact folder structure.
> Just extract it directly into `training/dataset/`.

**Verify your class names match `training/farm_obstacles.yaml`.**
Open the file and check the `names:` section:

```yaml
nc: 7
names:
  0: person
  1: rock
  2: tractor
  3: tree
  4: irrigation_pipe
  5: animal
  6: debris
```

If you added or renamed classes in Roboflow, update this file to match exactly.

---

## Step 4 — Train the Model

```powershell
# On CPU (any machine — takes ~30 min for 50 epochs)
python training/train.py --epochs 50

# On NVIDIA GPU (recommended — takes ~5 min for 100 epochs)
python training/train.py --epochs 100 --device cuda --batch 16

# On Apple Silicon Mac
python training/train.py --epochs 100 --device mps --batch 16
```

**What you'll see:**
```
=== Farm Obstacle Training ===
  Base model : yolov8n.pt
  Dataset    : training/farm_obstacles.yaml
  Epochs     : 50
  Device     : cpu

  Training images : 350

Starting training...

Epoch   1/50:  loss=3.42  mAP50=0.12
Epoch  10/50:  loss=1.87  mAP50=0.41
Epoch  25/50:  loss=0.94  mAP50=0.68
Epoch  50/50:  loss=0.71  mAP50=0.79

=== Training complete ===
Best weights saved to: runs/detect/farm_obstacles/weights/best.pt
```

**Reading the numbers:**
- `mAP50` = accuracy score (0 = nothing detected, 1.0 = perfect)
- `> 0.70` is good for a farm robot
- `> 0.80` is excellent

---

## Step 5 — Switch to Your Custom Model

Open `config/default.yaml` and change the `model` line:

```yaml
detectors:
  yolo:
    enabled: true
    model: "runs/detect/farm_obstacles/weights/best.pt"   # ← change this line
    confidence: 0.50
    danger_zone_ratio: 0.12
    device: "cpu"
```

Run the system:

```powershell
python main.py
```

The robot will now detect rocks, tractors, irrigation pipes and all other classes
you trained — not just COCO objects.

---

## Improving Accuracy Over Time

Good models are built through iteration. After deploying:

1. **Enable recording** in `config/default.yaml`:
   ```yaml
   recording:
     enabled: true
   ```

2. **Run in the field** — collect 30–60 min of real footage

3. **Review recordings** in `recordings/` — look for:
   - Missed detections (obstacle present but not detected)
   - False positives (empty ground flagged as obstacle)

4. **Add hard examples back to Roboflow** — upload the missed/wrong frames,
   re-label them, regenerate and re-export the dataset

5. **Re-train** — the new images join the existing dataset:
   ```powershell
   python training/train.py --epochs 50
   ```

6. **Repeat** — after 3–4 rounds the model learns the exact lighting, soil colour,
   and obstacle appearances of your specific farm

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Add `--batch 8` or `--batch 4` |
| `mAP50` stays below 0.3 | Need more images — aim for 100+ per class |
| Detects wrong class | Class names in YAML don't match Roboflow labels |
| Model file not found | Check the path in `config/default.yaml` exactly |
| Still shows only "person" | You're still using `yolov8n.pt` — update config to point to `best.pt` |
| False positives on soil | Add 20 more empty-field images and retrain |

---

## Quick Reference — All Commands

```powershell
# 1. Collect images (live camera)
python tools/collect_training_data.py

# 2. (Label in Roboflow — browser step)

# 3. Verify dataset structure
dir training\dataset\images\train\
dir training\dataset\labels\train\

# 4. Train
python training/train.py --epochs 50

# 5. Run with custom model (after updating config/default.yaml)
python main.py

# Run in simulation (no camera needed)
python main.py --simulate
```

---

## File Reference

| File | Purpose |
|------|---------|
| `tools/collect_training_data.py` | Live camera capture tool |
| `training/farm_obstacles.yaml` | Dataset config (class names + paths) |
| `training/train.py` | YOLOv8 fine-tuning script |
| `detectors/yolo_detector.py` | Inference engine (auto-detects custom vs pretrained) |
| `config/default.yaml` | Set `model:` path here to switch models |
| `runs/detect/farm_obstacles/weights/best.pt` | Your trained model (generated after training) |
