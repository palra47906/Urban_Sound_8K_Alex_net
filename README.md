
---

# 🎧 UrbanSound8K Audio Classification using AlexNet

This repository presents a robust and CPU-compatible implementation of the **AlexNet convolutional neural network** for environmental sound classification using the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). The model is trained exclusively on **Mel-spectrogram features**, with **no audio augmentation**, making it a strong baseline for research or academic study.

---

## 📌 Project Highlights

- 🔍 **Dataset**: UrbanSound8K (10 sound classes)
- 🧠 **Model Architecture**: AlexNet (adapted for Mel-spectrogram input)
- 💻 **Hardware Compatibility**: Runs efficiently on CPU (no GPU required)
- ❌ **No Audio Augmentation**: Ensures purity of learning from raw Mel features
- 📊 **Metrics Tracked**: Train, Validation, and Test Accuracy, Confusion Matrix, Classification Report

---

## 🎼 Dataset Overview

**UrbanSound8K** consists of 8732 sound clips (≤4 seconds) across **10 urban sound classes**:
```
air_conditioner, car_horn, children_playing, dog_bark, drilling,
engine_idling, gun_shot, jackhammer, siren, street_music
```

---

## 📁 Directory Setup

Ensure your directory contains:
```
/UrbanSound8K
├── audio/
│   ├── fold1/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

Update the paths in the code:
```python
DATA_PATH = '/content/drive/MyDrive/UrbanSound8K/audio/'
METADATA_PATH = '/content/drive/MyDrive/UrbanSound8K/metadata/UrbanSound8K.csv'
```

---

## 🧠 AlexNet Architecture Summary

Adapted to handle `168x168x1` Mel-spectrogram inputs:

```
Input (168x168x1)
→ Conv2D(96, 11x11, stride=4) → ReLU
→ MaxPooling(3x3, stride=2)
→ Conv2D(256, 5x5) → ReLU
→ MaxPooling(3x3, stride=2)
→ Conv2D(384, 3x3) → ReLU
→ Conv2D(384, 3x3) → ReLU
→ Conv2D(256, 3x3) → ReLU
→ MaxPooling(3x3, stride=2)
→ Flatten
→ Dense(4096) → ReLU → Dropout
→ Dense(4096) → ReLU → Dropout
→ Output Layer (Softmax)
```

---

## ⚙️ Preprocessing Workflow

1. Load metadata and audio files
2. Extract Mel-spectrograms using Librosa
3. Normalize spectrograms (Z-score)
4. Zero-pad or truncate to fixed shape (168x168)
5. One-hot encode class labels

---

## 🧪 Training Configuration

| Parameter         | Value             |
|------------------|------------------|
| Epochs           | 30               |
| Batch Size       | 32               |
| Optimizer        | Adam             |
| Loss             | Categorical Crossentropy |
| Early Stopping   | Patience = 10    |

---

## 📈 Results

| Metric              | Value (Approx.) |
|---------------------|-----------------|
| Train Accuracy      | ~99%            |
| Validation Accuracy | ~86%            |
| Test Accuracy       | ~85%            |

---

## 🔬 Evaluation Tools

- ✅ Accuracy (Train, Validation, Test)
- 📉 Loss curves
- 📊 Confusion Matrix
- 🧾 Classification Report (Precision, Recall, F1-score)

---

## 💻 Environment Requirements

```bash
pip install numpy pandas librosa matplotlib scikit-learn tensorflow
```

---

## 🚀 Getting Started

1. Download and extract the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
2. Clone this repository and open the notebook
3. Update paths for audio and metadata
4. Run all cells to train and evaluate the model

---

## 📚 Citation

If you use this work in academic research, please consider citing the UrbanSound8K dataset and AlexNet architecture.

---

