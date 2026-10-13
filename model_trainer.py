import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os
import pickle

# --- Configuration ---
DATA_FILE = 'data/posture_data.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'posture_lstm_model.h5')
SEQUENCE_LENGTH = 15 
# Full 33 key points: 33 * 3 = 99 features per frame
KEYPOINT_INDICES = list(range(33)) 

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise ValueError("The CSV file is empty. Run 1_data_collector.py first.")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please run 1_data_collector.py first.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()

# Separate features (X) and labels (Y)
X = df.drop('Label', axis=1).values
if X.shape[1] != len(KEYPOINT_INDICES) * 3:
    print(f"\nERROR: Feature count mismatch. Expected {len(KEYPOINT_INDICES) * 3} columns, found {X.shape[1]}.")
    print("Please delete data/posture_data.csv and re-run 1_data_collector.py.")
    exit()
Y = df['Label'].values


# --- 2. Scaling and Encoding ---
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Encoding labels...")
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y_categorical = to_categorical(Y_encoded)

NUM_CLASSES = len(label_encoder.classes_)
print(f"Detected Posture Classes: {label_encoder.classes_}")
if NUM_CLASSES != 3:
    print("\nERROR: Model requires exactly 3 classes (Good, Slouching, Forward_Head). Check data collection.")
    exit()

# --- 3. Create Time Series Sequences (Temporal Feature) ---
def create_sequences(X, Y, seq_length):
    X_sequences, Y_labels = [], []
    for i in range(len(X) - seq_length):
        seq = X[i:i + seq_length]
        Y_labels.append(Y[i + seq_length])
        X_sequences.append(seq)
    return np.array(X_sequences), np.array(Y_labels)

print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
X_seq, Y_seq = create_sequences(X_scaled, Y_categorical, SEQUENCE_LENGTH)

# --- 4. Train/Test Split ---
X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 5. Build and Train LSTM Model (Enhanced for robust data) ---
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape), 
        Dropout(0.3),
        LSTM(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]), num_classes=NUM_CLASSES)
model.summary()

# Use Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Starting model training...")
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=64, 
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping],
    verbose=1
)

# --- 6. Evaluation and Saving ---
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nModel Evaluation - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

model.save(MODEL_PATH)
print(f"Model successfully saved to {MODEL_PATH}")

# Save the scaler and encoder classes for use in the real-time application
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)
print("Scaler and Label Encoder saved for real-time use.")

