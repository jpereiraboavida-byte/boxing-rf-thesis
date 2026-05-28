# BAB 3: METODOLOJIA

## 3.1 Dezainu Peskiza

Peskiza ida-ne'e uza abordajen **peskiza desenvovimentu** (*research and development*) ne'ebé kombina:
- **Peskiza Kuantitativu:** Análiza estatístika markadór biokimiku no fíziku
- **Dezenvovimentu Sistema:** Implementa modelu machine learning no dashboard

## 3.2 Dataset Sintetiku

### 3.2.1 Justifikasaun Dataset Sintetiku
Dataset sintetiku escolhe tanba:
1. Dadus atleta real barak konfidensial
2. Bele kontrola distribuisaun ba eksperimentu kontroladu
3. Reprodus siensia desportu literatural bazeia ba parâmetru fisiologiku dokumentadu

### 3.2.2 Prosesu Jerasaun Dadus

**Konfigurasi Dataset:**
- Atleta: 30 ema (A001–A030)
- Semana: 20 (makrosiklu kompletu)
- Total rekorde: 30 × 20 = **600 rekorde**

**Parâmetru Fisiologiku per Faze:**

| Variável | GPP (μ±σ) | SPP (μ±σ) | Competition (μ±σ) | Taper (μ±σ) |
|----------|-----------|-----------|-------------------|-------------|
| Kortisol (nmol/L) | 580±60 | 540±55 | 490±50 | 450±45 |
| Testosterona (nmol/L) | 18±3 | 20±3 | 22±3 | 24±3 |
| CK (U/L) | 280±50 | 240±45 | 200±40 | 170±35 |
| HRV (ms) | 42±8 | 48±7 | 55±6 | 62±6 |
| VO2max (ml/kg/min) | 52±5 | 56±4 | 60±4 | 59±4 |
| Punch Speed (m/s) | 9.5±0.8 | 10.2±0.7 | 11.0±0.6 | 11.3±0.6 |
| Training Load | 72±10 | 65±10 | 50±10 | 35±8 |

### 3.2.3 Klasifikasaun Target (performance_level)

**Funsaun Skor:**
```
score = 0
if cortisol < 480 → score += 2  else if < 540 → score += 1
if hrv > 58      → score += 2  else if > 48  → score += 1
if vo2max > 58   → score += 2  else if > 53  → score += 1
if punch > 10.8  → score += 2  else if > 9.8 → score += 1
if endurance > 75→ score += 2  else if > 62  → score += 1
if testosterone > 22 → score += 1
if CK < 200          → score += 1
```

| Skor | Label |
|------|-------|
| ≥ 9  | Peak  |
| 5–8  | Medium|
| < 5  | Low   |

## 3.3 Pre-Prosesamentu Dadus

### 3.3.1 Feature Engineering

**Rasiu Testosterone/Cortisol:**
$$T/C = \frac{\text{testosterone}}{\text{cortisol} + \epsilon}$$

Rasiu T/C mak indikadór anaboliku/kataboliku validadu iha literatura siensia desportu.

**Recovery Score:**
$$RS = \left(\frac{HRV}{HRV_{max}} \times 0.4 + \left(1 - \frac{CK}{CK_{max}}\right) \times 0.3 + \left(1 - \frac{cortisol}{cortisol_{max}}\right) \times 0.3\right) \times 100$$

**Physical Composite Score:**
$$PCS = VO2max \times 0.3 + (1 - \frac{RT}{RT_{max}}) \times 100 \times 0.2 + PS \times 0.2 + End \times 0.15 + Str \times 0.15$$

**Cumulative Load:**
$$CL_i = \sum_{w=1}^{week_i} training\_load_w \quad \text{(per atleta)}$$

### 3.3.2 Encoding
- **Phase encoding:** GPP=0, SPP=1, Competition=2, Taper=3
- **Target encoding:** Low=0, Medium=1, Peak=2

### 3.3.3 Normalisasaun
StandardScaler (z-score normalization) aplika ba hotu-hotu feature numeriku:
$$z = \frac{x - \mu}{\sigma}$$

### 3.3.4 Data Splitting
- Train set: 80% (480 rekorde)
- Test set: 20% (120 rekorde)
- Stratifikasaun ba ekuilíbriu klas

## 3.4 Modelu Random Forest

### 3.4.1 Konfigurasi Modelu
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

### 3.4.2 Hyperparameter Tuning
Grid Search CV uza StratifiedKFold (k=5):
```
param_grid = {
    n_estimators:     [100, 200, 300]
    max_depth:        [None, 10, 20]
    min_samples_split:[2, 5]
    min_samples_leaf: [1, 2]
    max_features:     ["sqrt", "log2"]
}
```
**Métrika otimizasaun:** F1-Score (weighted)

## 3.5 Avaliasaun Modelu

| Métrika | Fórmula | Justifikasaun |
|---------|---------|---------------|
| Accuracy | TP+TN / Total | Métrika jerál |
| Precision | TP / (TP+FP) | Relevansia predisaun positiva |
| Recall | TP / (TP+FN) | Sensitividade deteksaun |
| F1-Score | 2×P×R / (P+R) | Balansa Precision-Recall |
| AUC-ROC | Área kurva ROC | Kapasidade diskriminasaun |

### 3.5.1 Cross-Validation
StratifiedKFold 5-fold uza ba estimativa generalizasaun robusta.

## 3.6 Dashboard Streamlit

Dashboard interativu inklui:
- **Tab EDA:** Distribusi target, statistika deskriptiva
- **Tab Tren:** Visualizasaun tren feature per semana
- **Tab Korelasi:** Heatmap korelasaun
- **Tab Modelu:** Rezultadu avaliasaun, feature importance, confusion matrix
- **Tab Predisaun:** Input slider ba atleta foun, predisaun real-time

## 3.7 Ferramenta no Teknolojia

| Kategoria | Teknolojia |
|-----------|-----------|
| Linguajen | Python 3.10+ |
| ML Library | scikit-learn 1.3.0 |
| Dadus | pandas 2.1.0, numpy 1.24.0 |
| Vizualizasaun | matplotlib, seaborn, plotly |
| Dashboard | Streamlit 1.28.0 |
| IDE | VS Code |
| Versaun kontrol | Git |
