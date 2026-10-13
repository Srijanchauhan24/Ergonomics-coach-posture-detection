import numpy as np

def extract_keypoints(results):
    if not results.pose_landmarks:
        return None

    coords = []
    for lm in results.pose_landmarks.landmark:
        coords.extend([lm.x * 1000, lm.y * 1000, lm.z * 1000])

    return np.array(coords).reshape(1, -1)


def update_sequence(history, new_data, seq_len):
    history.append(new_data[0])
    if len(history) > seq_len:
        history = history[-seq_len:]
    return history