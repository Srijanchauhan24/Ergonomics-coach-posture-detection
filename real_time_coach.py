import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
from tensorflow.keras.models import load_model

# --- Global Configuration and Setup ---
# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Model and Preprocessing Setup
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'posture_lstm_model.h5')
SEQUENCE_LENGTH = 15 

# COMPREHENSIVE KEY POINT INDICES - ALL 33 POINTS (0-32)
KEYPOINT_INDICES = list(range(33))

# Posture State Management
posture_history = []
bad_posture_duration = 0 
risk_level = 'LOW'
alert_message = ""
last_alert_time = 0
ALERT_COOLDOWN = 30 # Time between major alerts
FATIGUE_THRESHOLD_MODERATE = 60 * 2 # 2 minutes cumulative bad posture
FATIGUE_THRESHOLD_HIGH = 60 * 5   # 5 minutes cumulative bad posture

# --- 1. Load Model and Tools ---
try:
    model = load_model(MODEL_PATH)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    POSTURE_CLASSES = label_encoder.classes_
    print("AI Model and Preprocessing Tools loaded successfully.")
    if len(POSTURE_CLASSES) != 3:
        raise ValueError("Model was trained on an unexpected number of classes. Expected 3.")
except Exception as e:
    print(f"Error loading model/tools: {e}")
    print("Please run 1_data_collector.py and 2_model_trainer.py first to create the necessary files.")
    exit()

# --- 2. Feature Engineering Function ---
def get_keypoints_for_model(results):
    coords = []
    if not results.pose_landmarks:
        return None
        
    landmarks = results.pose_landmarks.landmark
    
    # Extract ALL 33 Coordinates 
    for i in KEYPOINT_INDICES:
        lm = landmarks[i]
        coords.extend([lm.x * 1000, lm.y * 1000, lm.z * 1000])

    return np.array(coords).reshape(1, -1)

# --- 3. Risk Assessment Engine ---
def update_risk_assessment(current_posture, frame_duration):
    global bad_posture_duration, risk_level, alert_message, last_alert_time
    current_time = time.time()
    
    is_bad_posture = (current_posture != 'Good')
    
    if is_bad_posture:
        bad_posture_duration += frame_duration
    else:
        # Allow the duration to decay quickly when posture is corrected
        bad_posture_duration = max(0, bad_posture_duration - frame_duration * 0.7) 

    # Determine Risk Level
    if bad_posture_duration < FATIGUE_THRESHOLD_MODERATE:
        risk_level = 'LOW'
        alert_color = (0, 255, 0) # Green
    elif bad_posture_duration < FATIGUE_THRESHOLD_HIGH:
        risk_level = 'MODERATE'
        alert_color = (0, 255, 255) # Yellow
    else:
        risk_level = 'HIGH'
        alert_color = (0, 0, 255) # Red

    # Generate Proactive Alerts
    if risk_level == 'HIGH' and (current_time - last_alert_time) > ALERT_COOLDOWN:
        
        if current_posture == 'Slouching':
            alert_message = "FATIGUE ALERT: Severe slouching detected! Stand up, stretch your back, and reset your seating."
        elif current_posture == 'Forward_Head':
            alert_message = "FATIGUE ALERT: Critical Forward Head. Adjust your screen height immediately! Pull your head back."
        else:
            alert_message = "FATIGUE ALERT: Cumulative strain is high. Take a mandatory 5-minute break and walk around!"
        last_alert_time = current_time
    
    elif risk_level == 'MODERATE' and (current_time - last_alert_time) > ALERT_COOLDOWN:
        alert_message = f"Warning: Posture is not optimal. Cumulative bad time: {int(bad_posture_duration)}s. Check your sitting/standing alignment."
        last_alert_time = current_time
    
    elif risk_level == 'LOW' and bad_posture_duration < 1:
        alert_message = "Posture is excellent. Keep up your focused work!"
        
    return alert_color

# --- 4. Main Real-Time Loop ---
def run_coach():
    global posture_history, alert_message, last_alert_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = time.time()
    
    # Set up pose detector
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            current_time = time.time()
            frame_duration = current_time - prev_time
            prev_time = current_time

            # --- Computer Vision Processing ---
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # --- Feature Engineering and Classification ---
            feature_vector = get_keypoints_for_model(results)
            current_posture = "Awaiting Detection"

            if feature_vector is not None:
                scaled_features = scaler.transform(feature_vector)
                
                posture_history.append(scaled_features[0])
                if len(posture_history) > SEQUENCE_LENGTH:
                    posture_history = posture_history[-SEQUENCE_LENGTH:]
                
                if len(posture_history) == SEQUENCE_LENGTH:
                    sequence = np.array(posture_history).reshape(1, SEQUENCE_LENGTH, -1)
                    prediction = model.predict(sequence, verbose=0)
                    predicted_class_index = np.argmax(prediction)
                    current_posture = POSTURE_CLASSES[predicted_class_index]
            
            # --- Risk Assessment and Alerts ---
            alert_color = update_risk_assessment(current_posture, frame_duration)

            # --- GUI Rendering and Overlay ---
            # 1. Draw Landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0) if current_posture == 'Good' else (0, 0, 255), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
            
            # 2. Display Status Panel
            cv2.rectangle(image, (0, 0), (640, 90), (10, 10, 10), -1) 
            
            # Posture Status
            cv2.putText(image, 'POSTURE:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, current_posture, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2, cv2.LINE_AA)
            
            # Risk Level
            cv2.putText(image, 'RISK LEVEL:', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, risk_level, (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2, cv2.LINE_AA)

            # Bad Posture Duration 
            cv2.putText(image, f'Bad Duration: {int(bad_posture_duration)}s', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)

            # 3. Display Alert Message Box
            if alert_message and (current_time - last_alert_time) < ALERT_COOLDOWN:
                cv2.rectangle(image, (0, image.shape[0] - 50), (640, image.shape[0]), alert_color, -1)
                cv2.putText(image, alert_message, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # Show the final output
            cv2.imshow('AI-Powered Posture Coach (Press Q to exit)', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_coach()
