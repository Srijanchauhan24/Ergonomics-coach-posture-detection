import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ================= CONFIG =================
DATA_FILE = 'data/posture_data.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'posture_model.keras')

SEQUENCE_LENGTH = 15
NUM_KEYPOINTS = 33   # Mediapipe full body
FEATURES_PER_FRAME = NUM_KEYPOINTS * 3  # 99

# Create model directory if not exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ================= LOAD DATA =================
print("Loading dataset...")

if not os.path.exists(DATA_FILE):
    print(f"ERROR: {DATA_FILE} not found. Run data collector first.")
    exit()

df = pd.read_csv(DATA_FILE)

if df.empty:
    print("ERROR: CSV file is empty.")
    exit()

# Separate features and labels
X = df.drop('Label', axis=1).values
Y = df['Label'].values

# Validate feature size
if X.shape[1] != FEATURES_PER_FRAME:
    print(f"ERROR: Expected {FEATURES_PER_FRAME} features, found {X.shape[1]}")
    print("Delete CSV and re-run data collection.")
    exit()

print(f"Dataset shape: {X.shape}")

# ================= PREPROCESSING =================
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Encoding labels...")
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y_categorical = to_categorical(Y_encoded)

classes = label_encoder.classes_
num_classes = len(classes)

print(f"Classes detected: {classes}")

if num_classes != 3:
    print("ERROR: Expected exactly 3 classes (Good, Slouching, Forward_Head)")
    exit()

# ================= CREATE SEQUENCES =================
def create_sequences(X, Y, seq_len):
    X_seq, Y_seq = [], []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        Y_seq.append(Y[i+seq_len])

    return np.array(X_seq), np.array(Y_seq)

print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
X_seq, Y_seq = create_sequences(X_scaled, Y_categorical, SEQUENCE_LENGTH)

print(f"Sequence shape: {X_seq.shape}")

# ================= TRAIN TEST SPLIT =================
X_train, X_test, Y_train, Y_test = train_test_split(
    X_seq, Y_seq, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# ================= BUILD MODEL =================
def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.3),

        LSTM(64, activation='relu'),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = build_model((SEQUENCE_LENGTH, X_train.shape[2]), num_classes)

model.summary()

# ================= TRAIN =================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("Training model...")

history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop],
    verbose=1
)

# ================= EVALUATE =================
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ================= SAVE =================
print("Saving model...")

# 🔥 IMPORTANT: Save in NEW FORMAT (fixes your error)
model.save(MODEL_PATH)

# Save scaler
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoder
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model saved at: {MODEL_PATH}")
print("Scaler & Label Encoder saved.")

print("\n✅ TRAINING COMPLETE")