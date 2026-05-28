# BAB 1: INTRODUSAUN

## 1.1 Kontestu no Motivasaun

Desportu boxe kompetisaun nian presiza planeamentu treinamentu ne'ebé sistematiku no bazia ba siensia atu atinji performansa maximu iha momentu kompetisaun. Periodizasaun treinamentu mak abordajen metodolojiku ida ne'ebé organiza karga treinamentu iha periudu tempu espesifiku atu otimiza adaptasaun atleta no mak oniku rekuperasaun biokimiku sira nian.

Iha era modernu, siensia desportu usa markadór biokimiku sira hanesan kortisol, testosterona, no Creatine Kinase (CK) atu monitoriza estadus rekuperasaun atleta. Variável sira ne'e, kombinadu ho indikadór performansa fíziku hanesan VO2max, tempu reaksaun, no velosidade sogu, bele fornese informasaun kompletu kona-ba pronti atleta ba kompetisaun.

## 1.2 Problema Peskiza

Problema sentrál ne'ebé teze ida-ne'e hatan mak:

> *"Oinsá maka bele uza algoritmu Random Forest Machine Learning atu predika nível performansa atleta boxe bazeia ba markadór biokimiku rekuperasaun no indikadór performansa fíziku iha periudu makrosiklu semana 20?"*

## 1.3 Objetivu Peskiza

### 1.3.1 Objetivu Jerál
Dezenvolve sistema predisaun performansa atleta boxe uza algoritmu Random Forest ne'ebé baziadu ba dadus biokimiku no fíziku iha makrosiklu semana 20.

### 1.3.2 Objetivu Espesifiku
1. Jera dataset sintetiku ne'ebé reprezenta realidade fisiologiku atleta boxe iha periudu treinamentu semana 20.
2. Implementa pre-prosesamentu dadus inklui feature engineering bazeia ba prinsipiu siensia desportu.
3. Treinu no otimiza modelu Random Forest Classifier atu klasifika nível performansa (Low / Medium / Peak).
4. Analiza importánsia feature hodi identifika markadór biokimiku no fíziku ne'ebé influénsia liu performansa.
5. Kria dashboard vizualizasaun interativu uza Streamlit.

## 1.4 Hipóteze Peskiza

**H1:** Markadór biokimiku (kortisol, HRV, testosterona) iha signifikánsia estatístika boot iha predisaun nível performansa atleta boxe.

**H2:** Rasiu Testosterone/Cortisol (T/C ratio) mak feature boot liu iha predisaun estadus rekuperasaun.

**H3:** Modelu Random Forest bele atinji akurásia ≥ 85% iha klasifikasaun nível performansa.

## 1.5 Skopus Peskiza

**Inklui:**
- Dataset sintetiku atleta boxe 30 ema iha periudu semana 20
- Variável biokimiku: kortisol, testosterona, CK, laktatu, HRV
- Variável fíziku: VO2max, tempu reaksaun, velosidade sogu, resisténsia, forsa
- Algoritmu: Random Forest Classifier

**La Inklui:**
- Dadus klíniki real atleta (konfidensialidade)
- Algoritmu machine learning seluk (SVM, Neural Network)
- Análiza biomekánika detalhadu

## 1.6 Benefísiu Peskiza

**Ba Prátiku Desportu:**
- Fó ferramenta objektivu ba treinador atu halo desizaun bazeia ba dadus
- Redus risku lesaun liu husi identifikasaun cedo sinál overtraining

**Ba Siénsia:**
- Demonstra aplikasaun machine learning iha kontekstu siensia desportu Timor-Leste
- Kontribuisaun ba literatura periodizasaun treinamentu

## 1.7 Estrutura Teze

| Bab | Konteúdu |
|-----|----------|
| Bab 1 | Introdusaun, objetivu, hipóteze |
| Bab 2 | Revisaun literatura: periodizasaun, biokimiku, Random Forest |
| Bab 3 | Metodolojia: dataset, pre-prosesamentu, modelu |
| Bab 4 | Rezultadu no diskusaun |
| Bab 5 | Konkluzaun no rekomendasaun |
