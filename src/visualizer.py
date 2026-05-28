"""
visualizer.py
Semua fungsi visualisasi untuk analisis dan evaluasi model
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs"
COLORS     = {"Low": "#e74c3c", "Medium": "#f39c12", "Peak": "#2ecc71"}
PHASE_COLORS = {"GPP": "#3498db", "SPP": "#9b59b6",
                "Competition": "#e67e22", "Taper": "#1abc9c"}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot disimpan: {path}")
    return path


# ── 1. Distribusi Target ──────────────────────────────────────────────────────
def plot_target_distribution(df):
    counts = df["performance_level"].value_counts()[["Low", "Medium", "Peak"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[k] for k in counts.index], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Distribusaun Performance Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance Level")
    ax.set_ylabel("Total Record")
    return _save(fig, "01_target_distribution.png")


# ── 2. Tren per Minggu ────────────────────────────────────────────────────────
def plot_weekly_trends(df, features=None):
    if features is None:
        features = ["cortisol_nmol_L", "hrv_ms", "vo2max_ml_kg_min",
                    "punch_speed_m_s", "endurance_score", "training_load"]
    weekly = df.groupby("week")[features].mean()
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(12, n * 2.5), sharex=True)
    for ax, feat in zip(axes, features):
        ax.plot(weekly.index, weekly[feat], marker="o", linewidth=2, markersize=4)
        ax.set_ylabel(feat.replace("_", " "), fontsize=9)
        ax.axvline(5.5,  color="#3498db", ls="--", lw=1, alpha=0.7, label="GPP→SPP")
        ax.axvline(12.5, color="#9b59b6", ls="--", lw=1, alpha=0.7, label="SPP→Comp")
        ax.axvline(17.5, color="#e67e22", ls="--", lw=1, alpha=0.7, label="Comp→Taper")
    axes[0].set_title("Tren Feature kada Semana (average all athlete)", fontweight="bold")
    axes[-1].set_xlabel("Semana")
    fig.tight_layout()
    return _save(fig, "02_weekly_trends.png")


# ── 3. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = [c for c in df.select_dtypes("number").columns
                        if c not in ["week", "performance_encoded", "phase_encoded"]]
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Semua Feature Numerik", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_correlation_heatmap.png")


# ── 4. Boxplot per Fase ───────────────────────────────────────────────────────
def plot_boxplot_by_phase(df, feature="hrv_ms"):
    phase_order = ["GPP", "SPP", "Competition", "Taper"]
    fig, ax = plt.subplots(figsize=(9, 6))
    data_by_phase = [df[df["phase"] == p][feature].dropna() for p in phase_order]
    bp = ax.boxplot(data_by_phase, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, phase in zip(bp["boxes"], phase_order):
        patch.set_facecolor(PHASE_COLORS[phase])
        patch.set_alpha(0.8)
    ax.set_xticklabels(phase_order)
    ax.set_title(f"Distribusi {feature} per Fase Training", fontweight="bold")
    ax.set_xlabel("Fase")
    ax.set_ylabel(feature.replace("_", " "))
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=p) for p in phase_order]
    ax.legend(handles=handles)
    return _save(fig, f"04_boxplot_{feature}.png")


# ── 5. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(feature_importance_series, top_n=15):
    fi = feature_importance_series.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi.index, fi.values,
                   color=sns.color_palette("viridis", len(fi)), edgecolor="white")
    for bar, val in zip(bars, fi.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_title(f"Top-{top_n} Feature Importance (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    return _save(fig, "05_feature_importance.png")


# ── 6. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm, labels=["Low", "Medium", "Peak"]):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Nilai Absolut)", "Confusion Matrix (Normalisasi)"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, cbar=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    return _save(fig, "06_confusion_matrix.png")


# ── 7. Performance per Fase ───────────────────────────────────────────────────
def plot_performance_by_phase(df):
    phase_order = ["GPP", "SPP", "Competition", "Taper"]
    perf_order  = ["Low", "Medium", "Peak"]
    grouped = (df.groupby(["phase", "performance_level"])
                 .size().unstack(fill_value=0)
                 .reindex(phase_order)[perf_order])
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="bar", ax=ax,
                 color=[COLORS[k] for k in perf_order],
                 edgecolor="white", width=0.6)
    ax.set_title("Distribusi Performance Level kada Fase Training", fontweight="bold")
    ax.set_xlabel("Fase")
    ax.set_ylabel("Jumlah Record")
    ax.set_xticklabels(phase_order, rotation=0)
    ax.legend(title="Performance Level")
    fig.tight_layout()
    return _save(fig, "07_performance_by_phase.png")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_generator import generate_dataset
    df = generate_dataset()
    plot_target_distribution(df)
    plot_weekly_trends(df)
    plot_correlation_heatmap(df)
    plot_boxplot_by_phase(df, "hrv_ms")
    plot_performance_by_phase(df)
    print("[OK] Semua plot selesai!")
