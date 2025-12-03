# Pneumonia Classification with CNN — Chest X‑Ray Dataset

This repository implements a Convolutional Neural Network (CNN) to classify chest X‑ray images (Normal vs Pneumonia) using the publicly available Chest X‑Ray Pneumonia dataset from Kaggle.  
It is intended as a baseline for experimentation, optimization, and comparative evaluation of different architectures or training regimes.

## Dataset

- The project uses the **Chest X‑Ray Images (Pneumonia)** dataset from Kaggle.
- The dataset contains approximately **5,863 JPEG chest‑X‑ray images** divided into two classes: **NORMAL** and **PNEUMONIA**.
- The directory structure is typically:

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

Because the full dataset is large and contains raw images, this repository does **not** include the images themselves — you need to download the dataset separately and place it under `chest_xray/` at the project root.

## Project Structure

```
.
├── pneumonia_cnn_baseline.ipynb
├── .gitignore
├── chest_xray/        # (NOT committed)
└── best_pneumonia_cnn.keras  # (NOT committed)
```

## Model and Workflow

- Baseline CNN using TensorFlow/Keras.
- Binary classification: NORMAL vs PNEUMONIA.
- Image preprocessing and optional augmentation.
- Saved model uses the modern `.keras` format.

## Setup & Usage

1. Clone this repository.
2. Download the dataset into `chest_xray/`.
3. (Optional) Create a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```
pip install tensorflow numpy matplotlib
```

5. Open and run the notebook:

```
jupyter notebook pneumonia_cnn_baseline.ipynb
```

6. Save/load model:

```
model.save("best_pneumonia_cnn.keras")
model = keras.models.load_model("best_pneumonia_cnn.keras")
```

## Notes

- CPU training takes ~12 min per epoch; GPU training is significantly faster.
- Dataset and model files are intentionally excluded from git.

## Future Work

- Try transfer learning (e.g., MobileNetV2, EfficientNet).
- Add early stopping, checkpoints, augmentation.
- Add evaluation metrics and confusion matrix.
