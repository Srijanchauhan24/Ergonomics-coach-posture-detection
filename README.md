# Ergonomics-coach-posture-detection
Real-time posture and ergonomics coach: MediaPipe pose landmarks + an LSTM classifier detect slouching and forward head posture, with a cumulative risk engine and voice alerts. Streamlit UI.
# 🧠 AI-Powered Posture & Ergonomics Coach

A real-time posture monitoring system that watches you through a webcam, classifies your posture with a deep learning model, and warns you when sustained bad posture builds up, before it turns into back or neck strain.

---

## How it works

1. **Pose estimation:** MediaPipe Pose extracts 33 body landmarks (x, y, z) per frame, giving 99 features per frame.
2. **Temporal classification:** A rolling window of 15 frames is standardized and fed to a stacked LSTM network, which classifies the posture as:
   - `Good`
   - `Slouching`
   - `Forward_Head`
3. **Risk engine:** Time spent in bad posture accumulates, and it decays faster when you correct yourself. Risk escalates in three levels:

   | Risk     | Cumulative bad-posture time |
   |----------|-----------------------------|
   | LOW      | under 2 minutes             |
   | MODERATE | 2 to 5 minutes              |
   | HIGH     | over 5 minutes              |

4. **Feedback:** On-screen overlays show posture, risk level and bad-posture duration. Spoken voice alerts (with a cooldown so they don't nag) tell you to fix your posture or take a break.

## Features

- Live webcam monitoring with a real-time posture and risk overlay
- Uploaded-video analysis with per-second posture reports
- Sequence-based model instead of single-frame classification, for steadier predictions
- Built-in data collection tool covering multiple scenarios: sitting frontal, sitting at 45°, standing frontal, and standing far from the camera
- Voice alerts via `pyttsx3`
- Two front ends: a Streamlit web app and a lightweight OpenCV desktop window

## Tech stack

Python · TensorFlow / Keras · MediaPipe · OpenCV · scikit-learn · Streamlit · pandas · NumPy · pyttsx3

## Model

| | |
|---|---|
| Input | 15 frames × 99 features (33 landmarks × xyz), standardized |
| Architecture | LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(3, softmax) |
| Loss / optimizer | Categorical cross-entropy / Adam |
| Training | 80/20 train-test split, early stopping on validation loss (patience 10) |

## Project structure

```
├── data_collector.py      # Record labeled pose data from the webcam
├── 2_model_trainer.py     # Train the LSTM; saves .keras model, scaler and label encoder
├── app.py                 # Streamlit app (live + video modes)
├── real_time_coach.py     # Standalone OpenCV real-time coach
├── utils.py               # Keypoint extraction and sequence buffering
├── risk_engine.py         # Cumulative risk scoring and alert logic
├── audio.py               # Voice alerts
├── requirements.txt
├── data/                  # Collected CSV dataset (created by data_collector.py)
└── models/                # posture_model.keras, scaler.pkl, label_encoder.pkl
```

## Getting started

### Prerequisites

- Python 3.9 to 3.11 (required by TensorFlow 2.15)
- A working webcam

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install tensorflow==2.15.0 mediapipe==0.10.9 opencv-python numpy==1.26.4 \
            pandas scikit-learn streamlit pyttsx3
```

### 1. Collect data

```bash
python data_collector.py
```

For each posture class (`Good`, `Slouching`, `Forward_Head`) and each scenario, press `s` to start a 10-second recording, or `q` to quit. Data is saved to `data/posture_data.csv`.

> If you want to re-collect data, delete the existing `data/posture_data.csv` first.

### 2. Train the model

```bash
python 2_model_trainer.py
```

This saves `posture_model.keras`, `scaler.pkl` and `label_encoder.pkl` into `models/`.

### 3. Run the coach

Web app:

```bash
streamlit run app.py
```

Or the OpenCV desktop window (press `Q` to quit):

```bash
python real_time_coach.py
```

> `real_time_coach.py` loads `models/posture_lstm_model.h5`. If you trained with `2_model_trainer.py`, change `MODEL_PATH` in that file to `models/posture_model.keras`.

## Limitations

- Accuracy depends on the data you collect; for best results, record data in your own setup (camera angle, lighting, desk).
- Classification covers three posture classes only.
- Works best with the upper body clearly visible to the camera.

## Future work

- Per-user calibration and personalized thresholds
- More posture classes (leaning, crossed legs, etc.)
- Break and stretch recommendations with session history analytics
- Browser-based inference via WebRTC

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
