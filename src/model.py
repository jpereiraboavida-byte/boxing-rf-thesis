"""
model.py
Training & evaluasi Random Forest Classifier
untuk prediksi performance_level atleta boxe
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)

LABEL_ORDER  = ["Low", "Medium", "Peak"]
MODEL_PATH   = "models/random_forest_model.joblib"
RESULTS_PATH = "outputs/model_results.txt"


# ── Training ──────────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train,
                        n_estimators=200,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    print(f"[OK] Random Forest trained | n_estimators={n_estimators}")
    return rf


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [None, 10, 20],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt", "log2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"[OK] Best params: {grid_search.best_params_}")
    print(f"     Best CV F1:   {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_


# ── Cross Validation ──────────────────────────────────────────────────────────
def cross_validate_model(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"[OK] Cross-Validation F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── Evaluasi ──────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, target_names=LABEL_ORDER)

    try:
        auc = roc_auc_score(
            pd.get_dummies(y_test).values, y_prob,
            multi_class="ovr", average="weighted"
        )
    except Exception:
        auc = None

    print(f"\n{'='*50}")
    print(f"  REJULTADU EVALUASI RANDOM FOREST")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc: print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{cr}")
    print(f"  Confusion Matrix:\n{cm}")

    results = {
        "accuracy": acc, "f1_weighted": f1, "auc_roc": auc,
        "confusion_matrix": cm, "classification_report": cr,
        "y_pred": y_pred, "y_prob": y_prob,
    }

    if feature_names is not None:
        fi = pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
        results["feature_importance"] = fi
        print(f"\n  Top-10 Feature Importance:")
        print(fi.head(10).to_string())

    return results


# ── Simpan & Muat Model ───────────────────────────────────────────────────────
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[OK] Model disimpan: {path}")


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"[OK] Model dimuat: {path}")
    return model


def save_results(results, path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("REJULTADU EVALUASI RANDOM FOREST\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy  : {results['accuracy']:.4f}\n")
        f.write(f"F1-Score  : {results['f1_weighted']:.4f}\n")
        if results['auc_roc']:
            f.write(f"AUC-ROC   : {results['auc_roc']:.4f}\n")
        f.write(f"\nClassification Report:\n{results['classification_report']}\n")
        f.write(f"Confusion Matrix:\n{results['confusion_matrix']}\n")
        if "feature_importance" in results:
            f.write(f"\nFeature Importance:\n{results['feature_importance'].to_string()}\n")
    print(f"[OK] Hasil disimpan: {path}")


if __name__ == "__main__":
    from preprocessor import run_pipeline
    X_train, X_test, y_train, y_test, features = run_pipeline()
    model = train_random_forest(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, features)
    cv_scores = cross_validate_model(model, X_train, y_train)
    save_model(model)
    save_results(results)
