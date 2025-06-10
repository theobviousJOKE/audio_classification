
# 🛠️ Sound-Based Drone Fault Classification using Multi-Microphone Data

This repository contains a complete pipeline for **multi-label audio classification** of drone flight data using **deep learning**. The system processes `.wav` audio files from multiple microphones to predict:

- ✅ **Maneuvering direction** (e.g., left, right, up)
- ⚙️ **Fault status** (e.g., normal, propeller fault, motor fault)
- ✈️ **Drone model type**

Audio features are extracted using MFCC, Chroma, and Mel spectrograms.

---

## 📂 Dataset

Dataset used:  
[Kaggle: Sound-Based Drone Fault Classification](https://www.kaggle.com/competitions/sound-based-drone-fault-classification-using-multi)

### Directory Structure:
```
/Dataset
    /drone_A
        /A
            /train
                /mic1
                /mic2
            /test
                /mic1
                /mic2
    /drone_B ...
    /drone_C ...
```

### File Naming Convention:
- **Model type:** character `0` (e.g., `"A"`)
- **Maneuvering direction:** character `2`
- **Fault type:** characters `4–7`

---

## 🚀 Project Pipeline

### 🔊 1. Feature Extraction
Each audio file is processed with `librosa`:
- **MFCCs**
- **Chroma STFT**
- **Mel Spectrogram**

Mean values of each feature are concatenated to form the final vector.

---

### 🧠 2. Multi-Output CNN Model

A **shared Conv1D CNN** backbone feeds into 3 separate dense output heads:
- **Main Output:** Maneuvering Direction
- **Auxiliary Output 1:** Fault Type
- **Auxiliary Output 2:** Drone Model Type

All heads use `softmax` with `categorical_crossentropy`.

---

### 📊 3. Training & Evaluation

- **Input:** Combined features from all drones/mics
- **Labels:** One-hot encoded (via `LabelEncoder + to_categorical`)
- **Training:** `Adam` optimizer, 20 epochs
- **Evaluation:** Accuracy reported for each of the 3 outputs

---

### 🧾 4. Submission Format

Predictions are saved to `submission1.csv`:

| ID            | model_type | maneuvering_direction | fault |
|---------------|------------|------------------------|-------|
| A_L_N.wav     | A          | L                      | N     |

---

## 📦 Requirements

Install dependencies:

```bash
pip install librosa tensorflow scikit-learn pandas
```

---

## ▶️ Running the Code

1. Make sure the dataset is available under the correct folder structure.
2. Run the script:  
   `audio_classification.py` (in Kaggle, Jupyter, or locally)
3. Check for:
   - Training metrics
   - Test accuracies
   - `submission1.csv` generated

---

## 🧠 Model Architecture Summary

```
Input → Conv1D + Pool → Conv1D + Pool → Conv1D + Pool → Flatten
   ├──→ Dense → Dropout → Output: Maneuver Direction
   ├──→ Dense → Dropout → Output: Fault Type
   └──→ Dense → Dropout → Output: Model Type
```

---

## 📈 Results (Sample)

| Task                    | Accuracy (Test Set) |
|-------------------------|---------------------|
| Maneuver Direction      | XX.XX%              |
| Fault Classification    | XX.XX%              |
| Model Type Classification | XX.XX%           |

> Replace `XX.XX%` after training the model.

---

## 📁 Folder Structure

```
audio_classification/
├── audio_classification.py
├── README.md
├── submission1.csv
└── Dataset/  # (not included due to size)
```

---

## ✍️ Author

**Swaroop Itkikar**  
Learning enthusiast, passionate about AI, Audio Processing, and Deep Learning.

---

## 📌 Notes

- You can further improve accuracy using spectrogram images + CNNs
- Data augmentation or ensemble models may help
- Pretrained audio models (like VGGish, OpenL3) can also be explored
