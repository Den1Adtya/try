"""
Script untuk melatih model Random Forest Klasifikasi Personality
(Introvert vs Extrovert) dan menyimpan model + encoder ke file .pkl
yang akan digunakan oleh aplikasi Streamlit.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("personality_dataset.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Encode Kolom Kategorikal
# -----------------------------
binary_cols = ["Stage_fear", "Drained_after_socializing"]
target_col = "Personality"

encoders = {}

for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # simpan supaya mapping (Yes/No -> 0/1) konsisten saat inference

target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])
encoders["Personality"] = target_encoder

print("\nMapping kolom biner:")
for col in binary_cols:
    print(f"  {col}: {dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))}")
print(f"  Personality: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

# -----------------------------
# 3. Split Fitur & Target
# -----------------------------
feature_order = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

X = df[feature_order]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 4. Train Random Forest (hyperparameter hasil tuning umum)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluasi
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi pada data uji: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# -----------------------------
# 6. Simpan Model & Encoder
# -----------------------------
artifact = {
    "model": model,
    "encoders": encoders,
    "feature_order": feature_order,
    "accuracy": acc,
}

with open("personality_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n✅ Model berhasil disimpan ke 'personality_model.pkl'")
