# Vehicle Detection Training:

## Project Overview:
This repository contains code, data splits, and trained weights for a vehicle detection project using the Ultralytics YOLO framework. A custom model `my_model.pt` was created from labeled images and saved in the workspace (see `my_model.pt` and `runs/detect/train2/weights/`).

---

## Dataset & Labeling:
- Total images: **462**
- Labeled with **Label Studio**: **132** images (exported to YOLO `*.txt` label format)
- Remaining images are available for further annotation to improve model performance.

---

## Methodology & Pipeline 🧭
1. Label images in **Label Studio** and export to YOLO format (one `.txt` file per image).
2. Use `train_val_split.py` to split labeled data into `data/train/` and `data/validation/` folders:

   `python train_val_split.py --datapath="/path/to/your/exported_data" --train_pct=0.8`

3. Train a YOLO model using the Ultralytics API or CLI. The repository contains training outputs under `runs/detect/train2/`.

---

## Optimization Techniques & Notes ⚡
- Transfer learning: start from a pretrained YOLO backbone (reduces needed labeled data).
- Data augmentation: enable random flips, mosaic, HSV jitter, and image scaling to improve generalization.
- Training tips for small datasets:
  - Increase epochs (e.g., 50–200) but use early stopping based on validation loss.
  - Use mixed precision (`fp16`) to reduce memory and speed up training if GPU supports it.
  - Use a smaller base model (e.g., `yolov8n`) and tune batch size (e.g., 8–16) to avoid overfitting.
- Save checkpoints frequently and monitor `runs/detect/<run>/results.csv` for metrics.

---

## Example Training (Ultralytics)
- Python API example:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or your chosen backbone
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)
```

- CLI example (if you have the Ultralytics CLI):

```
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

Adjust `data.yaml` to point to your `train/` and `validation/` image folders and class names.

---

## Usage: 
Install dependencies:

```
pip install -r requirements.txt
```

Run detection with the included script:

```
python yolodetect.py --model my_model.pt --source Inference-1.mp4 --resolution 1280x720
```

Useful options:
- `--thresh`: detection confidence threshold (default 0.5)
- `--record`: record processed output to `processed_output.avi`

Outputs:
- Tracking CSV log saved to `tracking_log.csv`
- Processed video (if `--record`) saved as `processed_output.avi`

---

## Files & Artifacts 📁
- `my_model.pt` — exported trained model (root folder)
- `my_model/` — additional model files and training run artifacts
- `runs/detect/train2/weights/best.pt` — best checkpoint from training
---