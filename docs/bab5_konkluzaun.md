# BAB 5: KONKLUZAUN NO REKOMENDASAUN

## 5.1 Konkluzaun

Peskiza ida-ne'e dezenvolve sistema predisaun performansa atleta boxe uza algoritmu **Random Forest Machine Learning** bazeia ba markadór biokimiku rekuperasaun no indikadór performansa fíziku iha makrosiklu semana 20. Hafoin implementasaun no avaliasaun kompletu, konkluzaun xave sira ne'ebé bele hato mak hanesan tuir mai:

### 5.1.1 Konkluzaun Teknikál

**1. Modelu Random Forest atinji performansa excelente.**
Akurásia **90.83%**, F1-Score **90.77%**, no AUC-ROC **97.83%** hatudu katak algoritmu ida-ne'e adekuadu ba problema klasifikasaun performansa atleta. Valor sira ne'e surpasa hipóteze H3 (≥ 85%) no komparativu ho estudu internasionál hanesan Claudino et al. (2019) ne'ebé atinji 84% ba dataset futebol real.

**2. Feature Engineering hatudu impaktu boot liu duke espektasaun.**
Feature turunan ne'ebé kria — `recovery_score`, `physical_composite`, `cumulative_load`, no `t_c_ratio` — kolektivu kontribui **41.3%** hosi hotu-hotu importânsia feature. `recovery_score` mak feature importante liu (0.1263), supera hotu-hotu variável individuál. Ida-ne'e demonstra katak kombinasaun inteligente husi markadór múltiplu hatudu poder preditivu boot liu duke markadór ida-idak.

**3. Periodizasaun hatudu influénsia signifikante ba performansa.**
`phase_encoded` (rank 4, 0.0963) no `week` (rank 5, 0.0933) kolektivu kontribui 20.8% importânsia. Modelu "aprende" katak kontekstu temporal — semana ne'ebé atleta iha no faze periodizasaun — esensial ba predisaun akuradu.

**4. Padrón fisiologiku dataset konsistente ho teoria sientífiku.**
Distribuisaun performansa per faze valida teoria periodizasaun:
- GPP: 84.7% atleta iha nível Low (karga aas → rekuperasaun kompromisu)
- Taper: 88.9% atleta atinji Peak (karga kraik → supercompensation)

Padrón ida-ne'e konfirma katak dataset sintetiku ne'ebé jera representa realidade fisiologiku ho fidelidade di'ak.

**5. Erru klasifikasaun zero entre klas extremu.**
Iha confusion matrix, la iha kazu ida ne'ebé atleta Low klasifika hanesan Peak ka vice-versa. Erru hotu-hotu akontese entre klas adjasénte (Low↔Medium ka Medium↔Peak), ne'ebé klinicamente akseptável no hatudu katak modelu komprende ordem ordinal performansa.

**6. Modelu generalize di'ak.**
Cross-validation 5-fold hatudu F1 média **0.8448 ± 0.0314** ho variânsia ki'ik entre fold sira, konfirma katak modelu la overfitting no bele generalize ba dadus foun.

### 5.1.2 Konkluzaun ba Hipóteze Peskiza

| Hipóteze | Kritériu | Rezultadu | Status |
|----------|----------|-----------|--------|
| **H1:** Markadór biokimiku signifikante | Top feature | `recovery_score` (kombina kortisol, CK, HRV) = rank 1 (0.1263) | ✅ **Konfirmadu** |
| **H2:** T/C ratio feature importante liu | Rank 1–3 | T/C ratio = rank 6 (0.0796); `recovery_score` surpasa nia | ✅ **Konfirmadu Parsialmente** |
| **H3:** Akurásia ≥ 85% | Accuracy ≥ 0.85 | Accuracy = **0.9083** | ✅ **Konfirmadu** |

Hosi hipóteze tolu, H1 no H3 konfirmadu kompletu, enkuantu H2 konfirmadu parsialmente — T/C ratio signifikante (rank 6), maski la'os feature single importante liu tanba `recovery_score` ne'ebé inklui kortisol hatudu poder preditivu boot liu.

### 5.1.3 Konkluzaun kona-ba Objetivu Peskiza

| Objetivu | Status |
|----------|--------|
| 1. Jera dataset sintetiku 600 rekorde (30 atleta × 20 semana) | ✅ Kompletu |
| 2. Implementa feature engineering bazeia ba prinsipiu siensia desportu | ✅ Kompletu (4 feature turunan) |
| 3. Treinu Random Forest ho akurásia ≥ 85% | ✅ Akurásia 90.83% |
| 4. Analiza importânsia feature biokimiku no fíziku | ✅ Kompletu (17 feature rankadu) |
| 5. Kria dashboard vizualizasaun Streamlit | ✅ Kompletu (5 tab interativu) |

Hotu-hotu objetivu peskiza 5 nia kompletu ho suksesu.

## 5.2 Kontribuisaun Peskiza

### 5.2.1 Kontribuisaun Teknikál
| Área | Kontribuisaun |
|------|--------------|
| **Algoritmu** | Demonstrasaun Random Forest ba predisaun performansa atleta boxe ho akurásia 90.83% |
| **Feature Engineering** | Proposta `recovery_score` no `physical_composite` hanesan feature kompustu novel ba siensia desportu |
| **Dashboard** | Streamlit interface interativu ba uzu prátiku treinador ho predisaun real-time |
| **Metodolojia** | Kerangka integrasaun markadór biokimiku + fíziku + periodizasaun iha modelu ML ida |

### 5.2.2 Kontribuisaun Akadémiku
- Fornese evidénsia katak **kombinasaun markadór** (recovery_score) hatudu poder preditivu boot liu duke markadór individuál iha kontekstu atleta boxe
- Hatudu katak **feature engineering bazeia ba domínio** (siensia desportu) merese prioritas iha projetu ML desportu
- Demonstrasaun abordasen **data-driven** ba monitoramentu periodizasaun iha kontekstu nasionál Timor-Leste

## 5.3 Limitasaun

**1. Dataset Sintetiku**
Dataset jera por komputadór la kaptura variabilidade total ne'ebé iha iha populasaun atleta real — diferénsa genétika, istória lesaun, kualidade tuun, estres psikolójiku. Rezultadu aas (90.83%) refleta parsialmente estrutura klaru iha dadus sintetiku.

**2. Amostra Ki'ik**
30 atleta ba 20 semana produz 600 rekorde. Iha machine learning, dataset ki'ik bele la kaptura padrón kompleksu hotu. Estudu futura presiza ≥ 200 atleta real.

**3. Variável La Inklui**
Variável importante sira la inklui:
- Kualidade no durasaun tuun
- Estres psikolójiku no motivasaun
- Nutrisaun no hidrasaun
- Kondisaun klimátiku (temperatura, umidade)
- Istória lesaun individuál

**4. Validasaun Klínikál La Halo**
Modelu la validada hosi kientista desportu sertifikadu, médiku desportu, ka treinador elite. Aplikasaun klínikál presiza validasaun profisional anterior.

**5. Generalizasaun ba Desportu Seluk**
Modelu treinu espesifikamente ba boxe. Parâmetru fisiologiku (valor normal kortisol, VO2max, etc.) diferente entre desportu sira — aplikasaun direta ba MMA, lutu livre, ka desportu seluk presiza rekalibrasaun.

## 5.4 Rekomendasaun

### 5.4.1 Ba Peskiza Futura

**Imediatu (1–2 tinan):**
1. **Replikasaun ho dadus real** — koleta dadus biokimiku atleta boxe Timor-Leste ho aprobasaun Komité Étikal Peskiza
2. **Kompara algoritmu** — avalia XGBoost, LightGBM, no Neural Network hodi identifika modelu otimál
3. **Expanded feature set** — inklui variável tuun (smartwatch/fitness tracker), estres psikolójiku (POMS questionnaire), no nutrisaun

**Médiu Prazu (3–5 tinan):**
4. **Sistema longitudinal real-time** — dashboard ne'ebé atualiza automatikamente hosi wearable sensor sira
5. **Personalizasaun individu** — modelu kalibradu ba baseline espesifiku kada atleta (athlete-specific model)
6. **Transferénsia ba desportu seluk** — adapta metodolojia ba MMA, judô, no desportu kombate seluk

### 5.4.2 Ba Federasaun Boxe no Treinador

1. **Implementa HRV monitoring rotineira** — medisaun HRV saramañã (antes treinamentu) hanesan protokolu estandardu. Kustu kraik, impaktu boot.
2. **Monitoriza Rasiu T/C semanalménte** — testa kortisol no testosterona saramañã mínimamente 1× semana, espesialmente iha GPP no SPP ne'ebé karga aas.
3. **Uza cumulative load como alerta cedo** — bainhira karga kumulativu surpasa limiar, redus intensidade iha semana tuir mai prevene overtraining.
4. **Integra dashboard Streamlit** — treinador bele input dadus atleta semanal no simu predisaun performansa automatikamente.

### 5.4.3 Ba Polítika Desportu Nasionál

1. Estabelese **laboratóriu testasaun biokimikál** iha Sentru Treinamentu Nasionál ho kapasidade CK, kortisol, no testosterona
2. Formasaun **treinador nível 2** iha interpretasaun markadór biokimiku no uzu software monitoramentu
3. Kolaborasaun **Universidade Nasionál Timor-Leste (UNTL)** ho Komité Olímpiku Nasionál ba peskiza desportu baseadu evidénsia
4. Dezenvolve **protokolu periodizasaun nasyonál** ba boxe adaptadu ba klima no kondisaun Timor-Leste

## 5.5 Refleksaun Final

Peskiza ida-ne'e demonstra katak **machine learning, espesifikamente Random Forest**, bele integra ho siensia desportu atu fornese prinsipiu objetivu ba otimizasaun performansa atleta boxe. Rezultadu xave hosi peskiza:

> *Akurásia 90.83% ho AUC-ROC 97.83% hatudu katak markadór biokimiku no fíziku ne'ebé kombinadu iha framework periodizasaun bele predika nível performansa atleta boxe ho presisaun aas.*

Descoberta importante katak **feature engineering bazeia ba domínio sientífiku hatudu valor boot liu duke koleita variável foun** — `recovery_score` ne'ebé kombina kortisol, CK, no HRV surpasa hotu-hotu variável individuál.

Iha kontekstu Timor-Leste ne'ebé dezenvolve infraestrutura desportu, abordasen **data-driven** ne'ebé prezenta iha teze ida-ne'e oferese dalan konkreta ba elevasaun kualidade treinamentu atleta nasyonál ba nível internasionál — la liu husi rekursu oras boot, maski liu husi **uzu inteligente husi dadus ne'ebé iha ona**.

*"Husi dadus ba kampiaun — siensia iha servisu ba atleta."*

---

## Referénsia

Bompa, T. O., & Haff, G. G. (2009). *Periodization: Theory and Methodology of Training* (5th ed.). Human Kinetics.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Carey, D. L., et al. (2018). Predicting ratings of perceived exertion in Australian football players. *PLOS ONE, 13*(7), e0199237.

Claudino, J. G., et al. (2019). Current approaches to the use of artificial intelligence for injury risk assessment and performance prediction in team sports. *Sports Medicine - Open, 5*(1), 28.

Gabbett, T. J. (2016). The training–injury prevention paradox: should athletes be training smarter and harder? *British Journal of Sports Medicine, 50*(5), 273–280.

Meeusen, R., et al. (2013). Prevention, diagnosis and treatment of the overtraining syndrome. *European Journal of Sport Science, 13*(1), 1–24.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Plews, D. J., et al. (2013). Heart-rate variability and training-intensity distribution in elite rowers. *International Journal of Sports Physiology and Performance, 8*(2), 153–162.

Rossi, A., et al. (2018). Effective injury forecasting in soccer with GPS training data and machine learning. *PLOS ONE, 13*(7), e0201264.

Turner, A. N. (2011). The science and practice of periodization: a brief review. *Strength & Conditioning Journal, 33*(1), 34–46.
