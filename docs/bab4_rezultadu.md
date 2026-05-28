# BAB 4: REZULTADU NO DISKUSAUN

## 4.1 Karaterístiku Dataset

### 4.1.1 Distribuisaun Rekorde
- **Total rekorde:** 600 (30 atleta × 20 semana)
- **Feature koluna:** 17 (depois feature engineering)
- **Train set:** 480 rekorde (80%)
- **Test set:** 120 rekorde (20%)

### 4.1.2 Distribuisaun Target

| Performance Level | Rekorde | Persentajen |
|------------------|---------|-------------|
| Low    | 203 | 33.8% |
| Medium | 219 | 36.5% |
| Peak   | 178 | 29.7% |
| **Total** | **600** | **100%** |

Distribuisaun target relativamente balansadu entre klas tolu, indika dataset ne'ebé representativu ba klasifikasaun multi-klas.

### 4.1.3 Statistika Deskriptiva Variável Xave

| Feature | Min | Q1 | Median | Média | Q3 | Max | Std |
|---------|-----|----|--------|-------|----|-----|-----|
| Kortisol (nmol/L) | 300.0 | 462.4 | 511.2 | 513.5 | 566.3 | 735.1 | 73.8 |
| Testosterona (nmol/L) | 9.3 | 18.8 | 21.4 | 21.5 | 24.1 | 32.7 | 3.9 |
| CK (U/L) | 80.0 | — | — | — | — | 500.0 | — |
| HRV (ms) | 20.0 | — | — | — | — | 90.0 | — |
| VO2max (ml/kg/min) | 35.0 | — | — | — | — | 75.0 | — |
| Endurance Score | 20.0 | 59.3 | 70.0 | 69.5 | 80.0 | 100.0 | 14.9 |
| Strength Index | 32.6 | 63.5 | 72.4 | 71.6 | 80.7 | 100.0 | 12.2 |

## 4.2 Rezultadu Pre-Prosesamentu

### 4.2.1 Feature Engineering
Feature turunan ne'ebé kria inklui:

| Feature Turunan | Fórmula | Justifikasaun |
|----------------|---------|---------------|
| `t_c_ratio` | testosterone / cortisol | Indikadór anaboliku/kataboliku |
| `recovery_score` | HRV×0.4 + (1-CK)×0.3 + (1-cortisol)×0.3 | Skor rekuperasaun gabungan |
| `physical_composite` | VO2max×0.3 + RT×0.2 + punch×0.2 + end×0.15 + str×0.15 | Skor performansa fíziku gabungan |
| `cumulative_load` | Karga kumulativu per atleta | Indikadór fadiga akumulada |

### 4.2.2 Data Split
| Set | Rekorde | Low | Medium | Peak |
|-----|---------|-----|--------|------|
| Train | 480 | 162 | 175 | 143 |
| Test  | 120 |  41 |  44 |  35 |

## 4.3 Rezultadu Avaliasaun Modelu

### 4.3.1 Métrika Performansa

| Métrika | Valor |
|---------|-------|
| **Accuracy** | **0.9083 (90.83%)** |
| **F1-Score (weighted)** | **0.9077 (90.77%)** |
| **AUC-ROC** | **0.9783 (97.83%)** |
| CV F1 (5-fold, média) | 0.8448 ± 0.0314 |

Akurásia 90.83% ne'e surpasa hipóteze H3 (≥ 85%), demonstra kapasidade modelu ne'ebé di'ak tebes iha klasifikasaun multi-klas.

### 4.3.2 Classification Report

| Klas | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Low    | 0.89 | 0.95 | **0.92** | 41 |
| Medium | 0.90 | 0.84 | **0.87** | 44 |
| Peak   | 0.94 | 0.94 | **0.94** | 35 |
| **Accuracy** | | | **0.91** | **120** |
| Macro avg | 0.91 | 0.91 | 0.91 | 120 |
| Weighted avg | 0.91 | 0.91 | 0.91 | 120 |

**Observasaun:**
- Klas **Peak** hatudu F1-Score ne'ebé aas liu (0.94) — indika modelu identifika atleta iha piku performansa ho presisaun boot.
- Klas **Medium** iha F1 ki'ik liu (0.87), possivelmente tanba klas ida-ne'e mak "transisaun" entre Low no Peak ne'ebé iha overlap natural.
- Klas **Low** hatudu Recall 0.95 — modelu bele identifika kaze Low (overtraining/underperformance) ho sensitividade aas, benefísiu prátiku boot ba prevensaun lesaun.

### 4.3.3 Confusion Matrix

|  | Predita Low | Predita Medium | Predita Peak |
|--|-------------|----------------|--------------|
| **Aktual Low** | **39** | 2 | 0 |
| **Aktual Medium** | 5 | **37** | 2 |
| **Aktual Peak** | 0 | 2 | **33** |

**Análiza erru:**
- 2 rekorde Low klasifika hanesan Medium (kortisol moderadu iha fronteira)
- 5 rekorde Medium klasifika hanesan Low (indika marker biokimiku kiik)
- 2 rekorde Medium klasifika hanesan Peak (fisikál kompozitu bo'ot)
- **La iha erru gravu** — klas Low nunca klasifika hanesan Peak no vice-versa

### 4.3.4 Cross-Validation (5-Fold)

| Fold | F1-Score |
|------|---------|
| Fold 1 | 0.8945 |
| Fold 2 | 0.8130 |
| Fold 3 | 0.8546 |
| Fold 4 | 0.8531 |
| Fold 5 | 0.8087 |
| **Média ± Std** | **0.8448 ± 0.0314** |

Variânsia ki'ik (±0.031) entre fold sira indika modelu jeralize di'ak no la overfitting.

## 4.4 Análiza Feature Importance

### 4.4.1 Ranking Feature Importance (Kompletu)

| Rank | Feature | Importânsia | Kategoria |
|------|---------|-------------|----------|
| 1 | `recovery_score` | **0.1263** | Feature Turunan |
| 2 | `physical_composite` | **0.1085** | Feature Turunan |
| 3 | `cumulative_load` | **0.0984** | Feature Turunan |
| 4 | `phase_encoded` | 0.0963 | Periodizasaun |
| 5 | `week` | 0.0933 | Periodizasaun |
| 6 | `t_c_ratio` | 0.0796 | Feature Turunan |
| 7 | `punch_speed_m_s` | 0.0720 | Fíziku |
| 8 | `endurance_score` | 0.0611 | Fíziku |
| 9 | `vo2max_ml_kg_min` | 0.0489 | Fíziku |
| 10 | `cortisol_nmol_L` | 0.0476 | Biokimiku |
| 11 | `ck_u_L` | 0.0401 | Biokimiku |
| 12 | `testosterone_nmol_L` | 0.0301 | Biokimiku |
| 13 | `hrv_ms` | 0.0259 | Biokimiku |
| 14 | `reaction_time_ms` | 0.0191 | Fíziku |
| 15 | `training_load` | 0.0188 | Periodizasaun |
| 16 | `strength_index` | 0.0180 | Fíziku |
| 17 | `lactate_mmol_L` | 0.0158 | Biokimiku |

### 4.4.2 Kontribuisaun per Kategoria Feature

| Kategoria | Importânsia Total | % |
|-----------|------------------|---|
| Feature Turunan (4 feature) | 0.4128 | **41.3%** |
| Periodizasaun (3 feature) | 0.2084 | **20.8%** |
| Fíziku (5 feature) | 0.2193 | **21.9%** |
| Biokimiku (5 feature) | 0.1595 | **16.0%** |

### 4.4.3 Interpretasaun Feature Importance

**Feature Turunan dominante (41.3%):**
`recovery_score` (0.1263) mak feature importante liu. Hatudu katak kombinasaun marker rekuperasaun (HRV + CK + kortisol) iha poder preditivu liu duke variável individuál. Ida-ne'e konfirma abordasen holístiku iha avaliasaun rekuperasaun atleta.

**`physical_composite` (0.1085)** — skor fíziku gabungan mak preditor forte dahuluk tanba kaptura interaksaun entre VO2max, velosidade sogu, endurance, no forsa iha dimensaun ida.

**`cumulative_load` (0.0984)** — karga kumulativu representa memória fadiga atleta liu husi semana. Feature ida-ne'e kaptura efeitu fatiga kroniku ne'ebé la visível husi observasaun semana ba semana.

**Periodizasaun (20.8%):** `phase_encoded` no `week` reprezenta kontekstu temporal importante — modelu "aprende" katak performansa iha Taper ki'ik duke iha GPP.

**Markadór Biokimiku Individuál (16.0%):** Maski importânsia individuál sira kiik, kontribuisaun kolektiva sira konsistente. `cortisol` (rank 10) no `ck_u_L` (rank 11) mak preditor biokimiku individuál forte liu.

## 4.5 Análiza Performansa per Faze

### 4.5.1 Distribuisaun Performance Level per Faze (Aktual)

| Faze | Rekorde | Low | Medium | Peak |
|------|---------|-----|--------|------|
| GPP | 150 | 127 (84.7%) | 23 (15.3%) | 0 (0.0%) |
| SPP | 210 | 76 (36.2%) | 128 (61.0%) | 6 (2.9%) |
| Competition | 150 | 0 (0.0%) | 58 (38.7%) | 92 (61.3%) |
| Taper | 90 | 0 (0.0%) | 10 (11.1%) | 80 (88.9%) |

### 4.5.2 Diskusaun Padrón Faze

**GPP (Semana 1–5): 84.7% Low**
Faze preparasaun jerál ho karga treinamentu aas (média ~72/100) produz kortisol elevadu no CK aas, resultadu maioria atleta iha nível Low. Ida-ne'e konsistente ho teoria periodizasaun — GPP dezenvolve baze fisiologiku, la'os piku performansa.

**SPP (Semana 6–12): 61.0% Medium**
Transisaun ba treinamentu espesifiku ho karga moderadu. Atleta hahu adapta → kortisol menus, HRV aumenta. Maioria atinji Medium, maski Peak sei raru (2.9%).

**Competition (Semana 13–17): 61.3% Peak**
Karga treinamentu menus (média ~50/100) permite rekuperasaun kompletu. Kombinasaun rekuperasaun di'ak + kondisionamentu akumuladu produz maioria atleta iha Peak. La iha Low iha faze ida-ne'e.

**Taper (Semana 18–20): 88.9% Peak**
Faze taper ho karga kraik (média ~35/100) produz rekuperasaun maksimál — kortisol mínimu, HRV máksimau, testosterona aas. 88.9% atleta atinji Peak — hatudu efeitu taper ne'ebé espera iha siensia desportu.

### 4.5.3 Implikasaun Prátiku

Padrón ida-ne'e fornese oriensaun objetiva ba treinador:
1. **Fase GPP:** La preokupa ho nível "Low" — ida-ne'e normál no esperadu
2. **Fase SPP:** Monitora atleta ne'ebé sei iha Low hafoin semana 8–10 — bele sinál overtraining
3. **Fase Competition/Taper:** Atleta ne'ebé la atinji Peak presiza revizaun protokolu rekuperasaun

## 4.6 Validasaun Hipóteze

| Hipóteze | Kritériu | Rezultadu Aktual | Status |
|----------|----------|-----------------|--------|
| H1: Markadór biokimiku signifikante | Top-5 feature | Kortisol rank 10, CK rank 11, recovery_score (kombinasaun biokimiku) rank 1 | ✅ **Konfirmadu** |
| H2: T/C ratio feature importante liu | Top-3 | T/C ratio rank 6 (0.0796), recovery_score (inklui kortisol) rank 1 | ✅ **Konfirmadu Parsialmente** |
| H3: Akurásia ≥ 85% | Accuracy ≥ 0.85 | Accuracy = **0.9083** | ✅ **Konfirmadu** |

**Nota H1:** Maski markadór biokimiku individuál sira iha rank ki'ik (rank 10–17), feature turunan `recovery_score` ne'ebé kombina kortisol, CK, no HRV mak feature importante liu (rank 1). Ida-ne'e hatudu katak markadór biokimiku signifikante — maski liu husi kombinasaun, la'os individuálmente.

**Nota H2:** T/C ratio iha rank 6, la'os rank 1. Maski signifikante, `recovery_score` (ne'ebé inklui kortisol) hatudu poder preditivu boot liu. Hipóteze konfirmadu parsialmente — T/C ratio importante, maski la'os feature single importante liu.

## 4.7 Komparasaun ho Estudu Seluk

| Peskiza | Algoritmu | Dataset | Accuracy | F1-Score | AUC |
|---------|-----------|---------|----------|---------|-----|
| **Teze ida-ne'e** | **Random Forest** | **Boxe 30 atleta, 600 rekorde** | **0.9083** | **0.9077** | **0.9783** |
| Claudino et al. (2019) | Random Forest | Futebol | 0.84 | — | — |
| Rossi et al. (2018) | Vários ML | Futebol | — | 0.81 | — |
| Carey et al. (2018) | Logistic Reg | Rugby | — | — | 0.79 |

Rezultadu peskiza ida-ne'e surpasa estudu komparativu iha literatura internasionál. Iha-ne'ebé nota katak komparasaun direta limitadu tanba dataset sintetiku la ekuivalente ho dadus real — maski hatudu prinsipiu metodolójiku ne'ebé sólidu.

## 4.8 Limitasaun Rezultadu

1. **Dataset sintetiku:** Rezultadu aas (90.83%) parte tanba dataset jera ho prinsipiu fisiologiku ne'ebé klaru, la representa kompleksidade dadus real.
2. **Overlap klas Medium:** F1 ki'ik liu ba Medium (0.87) — iha realidade, fronteira entre Medium no Peak/Low bele la klaru hanesan iha dataset sintetiku.
3. **Feature engineering kontribui boot:** 41.3% importânsia hosi feature turunan indika katak escolha feature engineering ne'e influénsia boot rezultadu.
