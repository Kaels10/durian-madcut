# Durian Maturity Assessment App

An offline desktop application for assessing durian maturity using a **YOLOv11** model.  
**Classes:** `mature` · `immature` · `damaged`  
**Platform:** Raspberry Pi (Raspberry Pi OS) + Python 3.9+

---

## Setup (Raspberry Pi)

```bash
# 1. Clone or copy this folder to your Pi
# 2. Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

## Run

```bash
python3 main.py
```

---

## Usage

1. **Settings** – Browse and load your `.pt` YOLOv11 model file (exported from Roboflow)
2. **Classify** – Load a single durian image to see bounding boxes and maturity labels
3. **Batch** – Select a folder of images; the app processes all and shows a results table
4. **Export** – Save results as a `.csv` file

---

## Model Setup

Export your trained model from Roboflow:
> **Roboflow → Your Project → Versions → Export Model → PyTorch (.pt)**

Then load the `.pt` file in the **Settings** panel of the app.

---

## Color Legend

| Color | Class |
|-------|-------|
| 🟢 Green | Mature |
| 🟡 Yellow | Immature |
| 🔴 Red | Damaged |

---

## Raspberry Pi Notes

- Tested on **Raspberry Pi 4** (4GB RAM recommended)
- Uses CPU inference — no GPU required
- For better performance, export model to **NCNN** format in Roboflow
