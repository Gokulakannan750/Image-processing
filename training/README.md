# Custom Farm Obstacle Training Guide

Train YOLOv8 to detect **rocks, tractors, trees, irrigation pipes** and any
other obstacle specific to your farm — in 4 steps.

---

## Step 1 — Collect images from your farm camera

Go to your field and run the collector tool:

```powershell
python tools/collect_training_data.py
```

**Controls in the window:**
- `SPACE` — save current frame
- `A` — toggle auto-capture every 2 seconds
- `Q` — quit

**Tips for good training data:**
- Capture each obstacle type from multiple distances (close, mid, far)
- Capture from different angles (straight-on, left, right)
- Include different lighting: morning sun, overcast, midday
- Aim for **50–100 images per class minimum**
- Include "empty field" images (no obstacles) to reduce false positives

Images are saved to `training/dataset/images/raw/`

---

## Step 2 — Label images with Roboflow (free)

1. Go to [roboflow.com](https://roboflow.com) and create a free account
2. Create a new project → **Object Detection**
3. Upload all images from `training/dataset/images/raw/`
4. Draw bounding boxes around each obstacle and assign a class label:
   - `rock`
   - `tractor`
   - `tree`
   - `irrigation_pipe`
   - `animal`
   - `debris`
   - `person`  ← keep this for safety
5. Use Roboflow's **Auto-Label** feature to speed up labeling
6. Once labeled, go to **Generate** → add augmentations (flip, crop, brightness)
7. Export as **YOLOv8** format → download the zip

---

## Step 3 — Prepare the dataset

Extract the downloaded zip into `training/dataset/`:

```
training/dataset/
    images/
        train/   ← ~80% of images
        val/     ← ~20% of images
        test/    ← optional
    labels/
        train/   ← .txt files matching each image
        val/
```

Then edit `training/farm_obstacles.yaml` to match your class names exactly
(same spelling as you used in Roboflow).

---

## Step 4 — Train

```powershell
# CPU (slower, works on any machine)
python training/train.py --epochs 50

# GPU (recommended if you have NVIDIA)
python training/train.py --epochs 100 --device cuda --batch 16
```

Training takes ~30 minutes on CPU for 50 epochs, ~5 minutes on a GPU.

When done, you'll see:
```
Best weights saved to: runs/detect/farm_obstacles/weights/best.pt
```

---

## Step 5 — Switch the detector to your custom model

Edit `config/default.yaml`:

```yaml
detectors:
  yolo:
    enabled: true
    model: "runs/detect/farm_obstacles/weights/best.pt"
    confidence: 0.50
    danger_zone_ratio: 0.12
    device: "cpu"
```

Run the system — it will now detect rocks, tractors, and everything you
trained it on instead of only COCO classes.

---

## Improving accuracy over time

- Run the system on your field → record sessions (`recording.enabled: true`)
- Review recorded frames for missed detections or false positives
- Add those frames back to Roboflow and re-label
- Re-run training — each round improves the model
- After 3–4 rounds you'll have a model tuned to your exact farm conditions
