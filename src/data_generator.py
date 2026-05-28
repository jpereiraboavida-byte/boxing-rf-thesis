"""
data_generator.py
Jera dadus sintetiku atleta boxe iha makrosiklu semana 20
Generate synthetic boxing athlete data for a 20-week macrocycle

Atleta: 30 (A001 - A030)
Semana: 20
Faze: GPP (1-5), SPP (6-12), Competition (13-17), Taper (18-20)
"""

import numpy as np
import pandas as pd
import os

SEED = 42
rng = np.random.default_rng(SEED)

N_ATHLETES  = 30
N_WEEKS     = 20
ATHLETE_IDS = [f"A{str(i).zfill(3)}" for i in range(1, N_ATHLETES + 1)]

PHASE_MAP = {
    **{w: "GPP"         for w in range(1,  6)},
    **{w: "SPP"         for w in range(6,  13)},
    **{w: "Competition" for w in range(13, 18)},
    **{w: "Taper"       for w in range(18, 21)},
}

PHASE_PARAMS = {
    "GPP": dict(
        cortisol=(580, 60), testosterone=(18, 3), ck=(280, 50),
        lactate=(4.5, 0.8), hrv=(42, 8),
        vo2max=(52, 5), reaction=(220, 15), punch=(9.5, 0.8),
        endurance=(55, 10), strength=(60, 10),
        training_load=(72, 10),
    ),
    "SPP": dict(
        cortisol=(540, 55), testosterone=(20, 3), ck=(240, 45),
        lactate=(4.0, 0.7), hrv=(48, 7),
        vo2max=(56, 4), reaction=(210, 12), punch=(10.2, 0.7),
        endurance=(65, 9), strength=(68, 9),
        training_load=(65, 10),
    ),
    "Competition": dict(
        cortisol=(490, 50), testosterone=(22, 3), ck=(200, 40),
        lactate=(3.5, 0.6), hrv=(55, 6),
        vo2max=(60, 4), reaction=(195, 10), punch=(11.0, 0.6),
        endurance=(78, 8), strength=(77, 8),
        training_load=(50, 10),
    ),
    "Taper": dict(
        cortisol=(450, 45), testosterone=(24, 3), ck=(170, 35),
        lactate=(3.0, 0.5), hrv=(62, 6),
        vo2max=(59, 4), reaction=(190, 10), punch=(11.3, 0.6),
        endurance=(82, 7), strength=(80, 7),
        training_load=(35, 8),
    ),
}


def _apply_athlete_bias(n_athletes):
    return {
        "cortisol_bias":     rng.normal(0, 15, n_athletes),
        "testosterone_bias": rng.normal(0, 2,  n_athletes),
        "ck_bias":           rng.normal(0, 20, n_athletes),
        "hrv_bias":          rng.normal(0, 5,  n_athletes),
        "vo2max_bias":       rng.normal(0, 3,  n_athletes),
        "reaction_bias":     rng.normal(0, 8,  n_athletes),
        "punch_bias":        rng.normal(0, 0.5,n_athletes),
        "endurance_bias":    rng.normal(0, 5,  n_athletes),
        "strength_bias":     rng.normal(0, 5,  n_athletes),
    }


def _classify_performance(row):
    score = 0
    if row["cortisol_nmol_L"] < 480:   score += 2
    elif row["cortisol_nmol_L"] < 540: score += 1
    if row["hrv_ms"] > 58:   score += 2
    elif row["hrv_ms"] > 48: score += 1
    if row["vo2max_ml_kg_min"] > 58:   score += 2
    elif row["vo2max_ml_kg_min"] > 53: score += 1
    if row["punch_speed_m_s"] > 10.8:  score += 2
    elif row["punch_speed_m_s"] > 9.8: score += 1
    if row["endurance_score"] > 75:   score += 2
    elif row["endurance_score"] > 62: score += 1
    if row["testosterone_nmol_L"] > 22: score += 1
    if row["ck_u_L"] < 200:            score += 1
    if score >= 9: return "Peak"
    elif score >= 5: return "Medium"
    else: return "Low"


def generate_dataset():
    athlete_bias = _apply_athlete_bias(N_ATHLETES)
    records = []

    for week in range(1, N_WEEKS + 1):
        phase  = PHASE_MAP[week]
        params = PHASE_PARAMS[phase]
        phase_weeks = [w for w, p in PHASE_MAP.items() if p == phase]
        progress    = (week - min(phase_weeks)) / max(len(phase_weeks) - 1, 1)

        for i, athlete_id in enumerate(ATHLETE_IDS):
            m  = params
            ab = athlete_bias

            cortisol     = float(rng.normal(m["cortisol"][0],     m["cortisol"][1])     + ab["cortisol_bias"][i]     - progress * 15)
            testosterone = float(rng.normal(m["testosterone"][0], m["testosterone"][1]) + ab["testosterone_bias"][i] + progress * 1.5)
            ck           = float(rng.normal(m["ck"][0],           m["ck"][1])           + ab["ck_bias"][i]           - progress * 20)
            lactate      = float(rng.normal(m["lactate"][0],      m["lactate"][1]))
            hrv          = float(rng.normal(m["hrv"][0],          m["hrv"][1])          + ab["hrv_bias"][i]          + progress * 5)
            vo2max       = float(rng.normal(m["vo2max"][0],       m["vo2max"][1])       + ab["vo2max_bias"][i]       + progress * 2)
            reaction     = float(rng.normal(m["reaction"][0],     m["reaction"][1])     + ab["reaction_bias"][i]     - progress * 8)
            punch        = float(rng.normal(m["punch"][0],        m["punch"][1])        + ab["punch_bias"][i]        + progress * 0.5)
            endurance    = float(rng.normal(m["endurance"][0],    m["endurance"][1])    + ab["endurance_bias"][i]    + progress * 3)
            strength     = float(rng.normal(m["strength"][0],     m["strength"][1])     + ab["strength_bias"][i]     + progress * 2)
            training_load= float(rng.normal(m["training_load"][0],m["training_load"][1]))

            cortisol     = max(300.0, min(800.0, cortisol))
            testosterone = max(8.0,  min(35.0,  testosterone))
            ck           = max(80.0, min(500.0, ck))
            lactate      = max(1.5,  min(8.0,   lactate))
            hrv          = max(20.0, min(90.0,  hrv))
            vo2max       = max(35.0, min(75.0,  vo2max))
            reaction     = max(150.0,min(300.0, reaction))
            punch        = max(6.0,  min(14.0,  punch))
            endurance    = max(20.0, min(100.0, endurance))
            strength     = max(20.0, min(100.0, strength))
            training_load= max(10.0, min(100.0, training_load))

            records.append({
                "athlete_id":          athlete_id,
                "week":                week,
                "phase":               phase,
                "training_load":       round(training_load, 1),
                "cortisol_nmol_L":     round(cortisol,      1),
                "testosterone_nmol_L": round(testosterone,  2),
                "ck_u_L":              round(ck,            1),
                "lactate_mmol_L":      round(lactate,       2),
                "hrv_ms":              round(hrv,           1),
                "vo2max_ml_kg_min":    round(vo2max,        1),
                "reaction_time_ms":    round(reaction,      1),
                "punch_speed_m_s":     round(punch,         2),
                "endurance_score":     round(endurance,     1),
                "strength_index":      round(strength,      1),
            })

    df = pd.DataFrame(records)
    df["performance_level"] = df.apply(_classify_performance, axis=1)
    return df


def save_raw_data(df, path="data/raw/boxing_dataset.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Dataset salva: {path}  |  Shape: {df.shape}")
    print(f"     Distribuisaun target:\n{df['performance_level'].value_counts()}")


if __name__ == "__main__":
    df = generate_dataset()
    save_raw_data(df)
    print(df.head(10).to_string())
