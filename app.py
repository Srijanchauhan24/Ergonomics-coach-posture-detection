import streamlit as st
import cv2
import numpy as np
import time
import pickle
from tensorflow.keras.models import load_model
import mediapipe as mp

from utils import extract_keypoints, update_sequence
from risk_engine import get_risk
from audio import speak

# Load model (NEW FORMAT)
model = load_model("models/posture_model.keras")

scaler = pickle.load(open("models/scaler.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
CLASSES = label_encoder.classes_

SEQ_LEN = 15

mp_pose = mp.solutions.pose

st.title("🧠 AI Posture & Ergonomics Coach")

mode = st.sidebar.selectbox("Mode", ["Live Monitoring", "Video Monitoring"])

frame_window = st.empty()
info_box = st.empty()

# ================= LIVE =================
if mode == "Live Monitoring":

    if st.button("Start Live"):
        cap = cv2.VideoCapture(0)

        history = []
        prev_time = time.time()

        with mp_pose.Pose() as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                curr = time.time()
                frame_time = curr - prev_time
                prev_time = curr

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                posture = "Detecting..."

                keypoints = extract_keypoints(results)

                if keypoints is not None:
                    scaled = scaler.transform(keypoints)
                    history = update_sequence(history, scaled, SEQ_LEN)

                    if len(history) == SEQ_LEN:
                        seq = np.array(history).reshape(1, SEQ_LEN, -1)
                        pred = model.predict(seq, verbose=0)
                        posture = CLASSES[np.argmax(pred)]

                risk, duration, alert = get_risk(posture, frame_time)

                if alert:
                    speak(alert)

                cv2.putText(frame, f"Posture: {posture}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.putText(frame, f"Risk: {risk}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                frame_window.image(frame, channels="BGR")

                info_box.markdown(f"""
                ### 📊 Live Report
                - Posture: **{posture}**
                - Risk: **{risk}**
                - Bad Duration: **{int(duration)} sec**
                """)

# ================= VIDEO =================
elif mode == "Video Monitoring":

    video_file = st.file_uploader("Upload Video", type=["mp4","avi"])

    if video_file:
        with open("temp.mp4","wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")
        history = []
        frame_count = 0

        with mp_pose.Pose() as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                posture = "Detecting..."

                keypoints = extract_keypoints(results)

                if keypoints is not None:
                    scaled = scaler.transform(keypoints)
                    history = update_sequence(history, scaled, SEQ_LEN)

                    if len(history) == SEQ_LEN:
                        seq = np.array(history).reshape(1, SEQ_LEN, -1)
                        pred = model.predict(seq, verbose=0)
                        posture = CLASSES[np.argmax(pred)]

                frame_count += 1

                if frame_count % 30 == 0:
                    st.write(f"Second {frame_count//30}: {posture}")

                frame_window.image(frame, channels="BGR")