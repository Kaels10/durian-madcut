# Durian Maturity Assessment App

Offline desktop app for assessing durian maturity with **bounding boxes** and **side labels** on the live camera feed (or still images).  
**Classes:** `mature` · `immature` · `damaged`  

**Raspberry Pi 5:** use a **YOLO ONNX** model and **ONNX Runtime** (see below) — no PyTorch install on the Pi.  
**Windows / GPU PC:** you can load **`.pt`** if you install the optional PyTorch stack.

---

## Setup (Raspberry Pi OS, 64-bit)

```bash
chmod +x setup.sh
./setup.sh
```

## Run

```bash
python3 main.py
```

---

## Model export (important)

### On the Pi (recommended)

1. Train or download your YOLOv8 / YOLO11 weights (e.g. `best.pt`) on a PC.
2. Export to **ONNX** (Ultralytics CLI example):

   ```bash
   yolo export model=best.pt format=onnx imgsz=512 simplify
   ```

   Use an `imgsz` that matches what you set in the app **Settings** (default **512** on Pi, **640** elsewhere).

3. Copy `best.onnx` to the Pi and load it under **Settings**.

### On a PC with PyTorch (optional)

Install optional deps:

```bash
pip install -r requirements-torch.txt
```

Then load **`.pt`** in **Settings** (Roboflow: *Export → PyTorch*).

---

## Usage

1. **Settings** — Load **`.onnx`** (Pi) or **`.pt`** (PC). Adjust confidence and image size if needed.
2. **Camera** — Point the camera at a durian; boxes and **labels beside each fruit** update automatically.
3. **Classify** — Single image + overlay.
4. **Batch** — Folder of images and results table.
5. **Export** — Save batch results as **`.csv`**.

---

## Color legend

| Color | Class    |
|-------|----------|
| Green | Mature   |
| Yellow| Immature |
| Red   | Damaged  |

---

## Raspberry Pi notes

- **Pi 5**, **64-bit** OS, **4GB+ RAM** recommended.
- Default capture is **640×480** with **V4L2**; inference is **throttled** slightly so the UI stays responsive.
- If class names are missing from the ONNX file, the app assumes **mature → immature → damaged** for **3-class** models (class order must match training).
