import cv2
import mediapipe as mp
import csv
import numpy as np
import os
import time

# --- Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Set up directories
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
FINAL_CSV_FILE = os.path.join(DATA_DIR, 'posture_data.csv')

# POSTURE LABELS: The three final classification classes
POSTURE_LABELS = ['Good', 'Slouching', 'Forward_Head']

# Collection scenarios (Increased Robustness)
SCENARIOS = {
    'Sitting_Frontal': "Sit facing the camera at normal distance.",
    'Sitting_Side_Angle': "Sit at a 45-degree angle (side view) for detailed neck and back profile.",
    'Standing_Frontal': "Stand facing the camera (upper body visible).",
    'Standing_Far': "Stand far back so the whole body is visible (to test generalization/distance)."
}

# COMPREHENSIVE KEY POINT INDICES: ALL 33 MEDIAPIPE POINTS
# [0] to [32]
KEYPOINT_INDICES = list(range(33)) 

# CSV Header (33 points * 3 coords = 99 columns + 1 Label column)
csv_header = ['Label']
for i in KEYPOINT_INDICES:
    csv_header.extend([f'X{i}', f'Y{i}', f'Z{i}'])

# --- Keypoint Extraction Function ---
def get_keypoints(results):
    coords = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for i in KEYPOINT_INDICES:
            lm = landmarks[i]
            # Normalize coordinates to a common scale (0-1000)
            coords.extend([lm.x * 1000, lm.y * 1000, lm.z * 1000])
    if not coords:
        # Return 99 zeros if no points are detected
        return [0] * (len(KEYPOINT_INDICES) * 3)
    return coords

# --- Main Collector Loop ---
def collect_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    all_raw_data = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        for label in POSTURE_LABELS:
            print(f"\n==============================================")
            print(f"--- COLLECTING PRIMARY POSTURE: {label} ---")
            print(f"==============================================")
            
            for scenario, instruction in SCENARIOS.items():
                
                print(f"\nScenario: {scenario}. Instruction: {instruction}")
                print("Press 's' to start recording for 10 seconds. Press 'q' to quit.")
                
                # --- Waiting for User to start ---
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Display status
                    cv2.putText(image, f"READY: {label} ({scenario})", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, instruction, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Posture Data Collector', image)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('s'):
                        break
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # --- Start recording for 10 seconds ---
                start_time = time.time()
                duration = 10 
                print(f"Recording {label} ({scenario})...")
                
                while cap.isOpened() and (time.time() - start_time) < duration:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    keypoints_data = get_keypoints(results)
                    row = [label] + keypoints_data # Use the primary label
                    all_raw_data.append(row)

                    # Draw landmarks and timer
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    time_left = duration - (time.time() - start_time)
                    cv2.putText(image, f"RECORDING: {label} - {time_left:.1f}s", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Posture Data Collector', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                print(f"Finished recording for {label} in {scenario}. {len(all_raw_data)} total frames collected.")

    # --- Save final merged CSV ---
    print(f"\nFinal saved frames: {len(all_raw_data)}")
    with open(FINAL_CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(all_raw_data)

    cap.release()
    cv2.destroyAllWindows()
    print(f"All data successfully merged and saved to: {FINAL_CSV_FILE}")
    print("Data collection complete. Please run 2_model_trainer.py next.")

if __name__ == "__main__":
    collect_data()

