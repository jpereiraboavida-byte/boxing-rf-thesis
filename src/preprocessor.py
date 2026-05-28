"""
preprocessor.py
Preprocessing data atleta boxe:
- Load raw data
- Feature engineering
- Encoding & scaling
- Train/Test split
- Save processed data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURE_COLS = [
    "cortisol_nmol_L", "testosterone_nmol_L", "ck_u_L",
    "lactate_mmol_L", "hrv_ms",
    "vo2max_ml_kg_min", "reaction_time_ms", "punch_speed_m_s",
    "endurance_score", "strength_index",
    "training_load", "week",
]

TARGET_COL = "performance_level"
LABEL_ORDER = ["Low", "Medium", "Peak"]


def load_raw(path="data/raw/boxing_dataset.csv"):
    df = pd.read_csv(path)
    print(f"[OK] Raw data loaded: {df.shape}")
    return df


def feature_engineering(df):
    """Aumenta feature turunan nebee relevan secara fisiologis."""
    df = df.copy()

    # Rasio Testosterone/Cortisol → indikator anabolik vs katabolik
    df["t_c_ratio"] = df["testosterone_nmol_L"] / (df["cortisol_nmol_L"] + 1e-6)

    # Skor recovery gabungan (hrv tinggi, ck rendah, cortisol rendah)
    df["recovery_score"] = (
        (df["hrv_ms"] / df["hrv_ms"].max()) * 0.4
        + (1 - df["ck_u_L"] / df["ck_u_L"].max()) * 0.3
        + (1 - df["cortisol_nmol_L"] / df["cortisol_nmol_L"].max()) * 0.3
    ) * 100

    # Indeks performa fisik gabungan
    df["physical_composite"] = (
        df["vo2max_ml_kg_min"] * 0.3
        + (1 - df["reaction_time_ms"] / df["reaction_time_ms"].max()) * 100 * 0.2
        + df["punch_speed_m_s"] * 0.2
        + df["endurance_score"] * 0.15
        + df["strength_index"] * 0.15
    )

    # Beban kumulatif (penanda kelelahan)
    df = df.sort_values(["athlete_id", "week"]).reset_index(drop=True)
    df["cumulative_load"] = df.groupby("athlete_id")["training_load"].cumsum()

    print(f"[OK] Feature engineering selesai: {df.shape[1]} kolom")
    return df


def encode_and_scale(df, fit=True, scaler_path="models/scaler.joblib",
                     le_path="models/label_encoder.joblib"):
    df = df.copy()
    os.makedirs("models", exist_ok=True)

    # Encode phase
    df["phase_encoded"] = df["phase"].map(
        {"GPP": 0, "SPP": 1, "Competition": 2, "Taper": 3}
    )

    # Label encode target
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    df["performance_encoded"] = le.transform(df[TARGET_COL])

    # Feature list lengkap
    all_features = FEATURE_COLS + ["t_c_ratio", "recovery_score",
                                    "physical_composite", "cumulative_load",
                                    "phase_encoded"]

    scaler = StandardScaler()
    if fit:
        df[all_features] = scaler.fit_transform(df[all_features])
        joblib.dump(scaler, scaler_path)
        joblib.dump(le, le_path)
        print(f"[OK] Scaler & LabelEncoder disimpan ke models/")
    else:
        scaler = joblib.load(scaler_path)
        df[all_features] = scaler.transform(df[all_features])

    return df, all_features


def split_data(df, all_features, test_size=0.2, random_state=42):
    X = df[all_features]
    y = df["performance_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[OK] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"     Distribusi y_train: {dict(y_train.value_counts().sort_index())}")
    return X_train, X_test, y_train, y_test


def save_processed(df, path="data/processed/boxing_processed.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Processed data disimpan: {path}")


def run_pipeline(raw_path="data/raw/boxing_dataset.csv"):
    df = load_raw(raw_path)
    df = feature_engineering(df)
    df_enc, all_features = encode_and_scale(df, fit=True)
    save_processed(df_enc)
    X_train, X_test, y_train, y_test = split_data(df_enc, all_features)
    return X_train, X_test, y_train, y_test, all_features


if __name__ == "__main__":
    run_pipeline()
