# BAB 2: REVISAUN LITERATURA

## 2.1 Periodizasaun Treinamentu iha Desportu Boxe

### 2.1.1 Konseitu Periodizasaun
Periodizasaun mak prinsipiu sientífiku ida hodi organiza treinamentu atleta iha períodu tempu sistematiku atu atinji piku performansa iha tempu kompetisaun (Bompa & Haff, 2009). Konseitu ida-ne'e bazeia ba prinsipiu *supercompensation* — iha-ne'ebé korpu adapta ba stres treinamentu liu husi faze rekuperasaun.

### 2.1.2 Estrutura Makrosiklu Semana 20
Iha boxe, makrosiklu tipikamente divide ba faze hirik:

| Faze | Semana | Objetivu | Karga |
|------|--------|----------|-------|
| GPP (General Preparation) | 1–5 | Dezenvolve baze aeróbiku no forsa jerál | Aas (70–80%) |
| SPP (Specific Preparation) | 6–12 | Abilidade espesifiku boxe, teknika | Moderadu-aas (60–75%) |
| Competition | 13–17 | Piku performansa, taktika kompetisaun | Moderadu (45–60%) |
| Taper | 18–20 | Rekuperasaun, manutensaun piku | Kraik (30–40%) |

### 2.1.3 Prinsipiu Karga Treinamentu
Turner (2011) argumenta katak variasaun karga treinamentu sistematiku esensial atu prevene *overtraining syndrome* no maximiza adaptasaun fisiologiku. Karga aas demais produz aumentu kortisol kroniku ne'ebé suprime sistema imuniu no degrade performansa.

## 2.2 Markadór Biokimiku iha Monitoramentu Rekuperasaun

### 2.2.1 Kortisol
Kortisol mak hormona stres ne'ebé produz husi glándula adrenal iha resposta ba intensidade ezersísiu aas. Nível kortisol elevadu indika:
- Karga treinamentu aas
- Rekuperasaun inadekuadu
- Potensial *overreaching* ka *overtraining*

Nível normal kortisol iha atleta: 300–700 nmol/L (Meeusen et al., 2013).

### 2.2.2 Testosterona
Testosterona mak hormona anabolika prinsipál ne'ebé promove sínteze proteína muskulu. Nível aas indika estadus rekuperasaun bon no kapasidade adaptasaun treinamentu di'ak. Rasiu T/C (Testosterone/Cortisol) konsidera indikadór sensitivu estado anaboliku/kataboliku atleta.

### 2.2.3 Creatine Kinase (CK)
CK mak enzima muskular ne'ebé la'o ba sirkulasaun sanguínea bainhira iha estragu muskular. Nível CK elevadu hafoin sesaun treinamentu intensu indika estragu miofibrila boot liu — sinál katak rekuperasaun presiza tempu naruk liu.

### 2.2.4 Laktatu
Laktatu sérika mak produto metabolizmu anaeróbiku. Akumulasaun laktatu sanguínea relasiona ho intensidade ezersísiu no kapasidade bufer atleta. Atleta kondisonadu di'ak hatudu *lactate clearance* rápidu liu.

### 2.2.5 Heart Rate Variability (HRV)
HRV mak variasaun tempu entre batimentu kardíaku konsekutivu. HRV aas indika:
- Atividade sistema nervóza parasimpátiku dominante
- Rekuperasaun di'ak
- Pronti ba treinamentu intensu

HRV konsideradu iha literatura atual hanesan unu husi indikadór rekuperasaun objetivu ne'ebé konfiável liu (Plews et al., 2013).

## 2.3 Indikadór Performansa Fíziku iha Boxe

### 2.3.1 VO2max
VO2max reprezenta kapasidade maksimál konsumu oksijéniu durante ezersísiu, indikatór kardinál kapasidade aeróbiku atleta. Iha boxe, atleta elite hatudu VO2max iha intervalo 55–68 ml/kg/min.

### 2.3.2 Tempu Reaksaun
Tempu reaksaun kritikal ba boxe — atleta ho tempu reaksaun kraik (bele inisiativa ka respondé rápidu) iha vantajen tátiku signifikativu. Training abilidade motorsensorial bele redus tempu reaksaun husi 220ms ba 185ms iha atleta elite.

### 2.3.3 Velosidade Sogu
Velosidade sogu (punch speed) determina forsa impaktu no efektividade atake. Atleta elite hatudu velosidade 9–12 m/s dependente ba kategoría pezu no teknika.

## 2.4 Random Forest Machine Learning

### 2.4.1 Introdusaun Random Forest
Random Forest (Breiman, 2001) mak algoritmu ensemble ne'ebé kombina multiple decision trees. Prinsipiu báziku:
1. Kria subset random husi dadus treinu (bootstrap sampling)
2. Treinu decision tree ba kada subset
3. Agrega predisaun hosi árvore sira (voting ba klasifikasaun)

### 2.4.2 Vantajen Random Forest
- **Robusteza:** Resistente ba overfitting kompara ho decision tree individual
- **Feature Importance:** Fornese estimativa importánsia feature automátiku
- **Manuseiu dados mista:** Bele maneia feature numeriku no kategóriku
- **Performansa:** Geralmente kompetitivu iha datasets tabelados

### 2.4.3 Aplikasaun iha Siensia Desportu
Estudus resente demonstra aplikasaun Random Forest iha:
- Predisaun lesaun atleta (Rossi et al., 2018)
- Klasifikasaun fadiga akuta vs króniku (Claudino et al., 2019)
- Otimizasaun karga treinamentu (Gabbett, 2016)

### 2.4.4 Hiperparâmetru Xave
| Hiperparâmetru | Deskrisaun | Tipiku |
|---------------|------------|--------|
| n_estimators | Nú. árvore | 100–500 |
| max_depth | Profundidade máxima árvore | 10–30 ka None |
| min_samples_split | Amostra mínima atu divide nó | 2–10 |
| max_features | Feature sira kandidatu per split | sqrt, log2 |

## 2.5 Peskiza Relevante Anterior

| Autór | Tópiku | Rezultadu |
|-------|--------|-----------|
| Claudino et al. (2019) | RF ba predisaun lesaun futebol | Akurásia 84% |
| Rossi et al. (2018) | ML ba monitoramentu atleta | F1=0.81 |
| Carey et al. (2018) | Load monitoring Rugby | AUC=0.79 |

## 2.6 Kerangka Teóriku Peskiza

```
Dadus Biokimiku      Dadus Fíziku
(Kortisol, HRV,   +  (VO2max, Punch Speed,  ──► Feature Engineering
 CK, Testosterona,    Reaction Time, etc.)        (T/C ratio, Recovery
 Laktatu)                                          Score, etc.)
                                                        │
                                              Random Forest Classifier
                                                        │
                                         Performance Level: Low/Medium/Peak
```
