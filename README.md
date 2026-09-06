<div align="center">

# 🧠 EpilepsyDetection

**EEG-based epilepsy classification with machine learning, from research notebook to clinical web demo.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

[Quick start](#-quick-start) •
[Dataset](#-dataset) •
[Analysis](#-notebook-analysis) •
[Findings](#-key-findings) •
[Web app](#-kymapsis-web-app) •
[API](#endpoints) •
[Limitations](#%EF%B8%8F-notes-and-limitations)

</div>

---

## ✨ What is this?

Epileptic patients in a seizure-free interval still show subtle abnormalities in their EEG. This project trains a model that tells apart, within the **same patient**, a segment recorded **on the epileptogenic zone** (Level 2) from one recorded **on the opposite, healthy hemisphere** (Level 3). It ships two things:

| | |
|---|---|
| 📓 **Research notebook** | Full pipeline on the Andrzejak (Bonn) EEG dataset: outlier analysis, model selection, statistical comparison, reduced-window and frequency-domain experiments. |
| 🏥 **Kymapsis web demo** | A Flask app simulating a healthcare portal. Doctors upload a trace, the trained Random Forest classifies it, and a report is sent to the patient portal. |

```
EEG-data.xlsx ──► EEG_final.ipynb ──► modello_epilessia.pkl ──► app_flask (Kymapsis)
   dataset        analysis + training     Random Forest           web portal
```

## 🚀 Quick start

```bash
git clone <repo-url>
cd EpilepsyDetection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask pandas numpy scikit-learn joblib
cd app_flask && python app.py
```

You will be prompted for which portal to open:

```
Quale versione vuoi aprire? (p = paziente, d = dottore):
```

Type `d` for the **doctor** portal or `p` for the **patient** portal. The browser opens automatically on `http://127.0.0.1:5002`. To try a prediction, open a patient record from the doctor portal and upload `test_patient.csv`.

<details>
<summary><b>Run the notebook too</b></summary>

```bash
pip install jupyter matplotlib seaborn scipy scikit-posthocs mlxtend openpyxl
jupyter notebook EEG_final.ipynb
```

Run the cells sequentially. `EEG-data.xlsx` must sit next to the notebook.

</details>

## 📊 Dataset

`EEG-data.xlsx` holds **500 single-channel EEG recordings**, each 23.6 s long and sampled into **4096 points** (columns `X1` … `X4096`, ≈ 173.61 Hz). The `y` column is the class:

| Level | Subject | Condition | Site |
|:-:|---|---|---|
| 1 | Epileptic | Seizure activity | Epileptogenic zone |
| **2** | Epileptic | Seizure-free interval | **Epileptogenic zone** |
| **3** | Epileptic | Seizure-free interval | **Opposite hemisphere** |
| 4 | Healthy | Eyes open | Scalp |
| 5 | Healthy | Eyes closed | Scalp |

The analysis and the shipped model target the binary task **Level 2 vs. Level 3**.

## 🔬 Notebook analysis

`EEG_final.ipynb` walks through six stages:

1. **Dataset exploration**: load the data and plot sample signals for each level.
2. **Outlier analysis**: standardize signals per class and flag patients with an excessive share of extreme values (> 2σ). Produces a raw and an outlier-cleaned dataset for comparison.
3. **Model selection with GridSearchCV**: logistic regression baseline, then SVM (linear, polynomial, RBF) and Random Forest tuned with stratified k-fold. Evaluated with ROC curves, confusion matrices and classification reports on both datasets.
4. **Statistical model comparison**: 5x2cv paired t-test, majority and soft voting ensembles, AdaBoost and stacking, Friedman test with Nemenyi post-hoc and critical-distance plots.
5. **Reduced time window**: classify using only half the recording (11.8 s) to assess whether shorter acquisitions are clinically viable.
6. **Frequency domain**: FFT preprocessing and Welch power spectral density across the canonical EEG bands (Delta, Theta, Alpha, …).

## 🏆 Key findings

> **Outliers are signal, not noise.** Extreme peaks encode epileptic patterns. SVM (RBF and polynomial kernels) exploits them as support vectors and performs best on the **raw** dataset, whereas Random Forest overfits to them and benefits from **outlier removal**.

> **Half the recording is enough.** With 11.8 s instead of 23.6 s, performance is nearly unchanged: accuracy ≈ 0.886, AUC ≈ 0.982. Shorter EEG sessions look clinically feasible.

> **Differences are tested, not eyeballed.** Every model comparison is backed by 5x2cv and Friedman/Nemenyi tests to separate real gains from random fluctuation.

## 🏥 Kymapsis web app

`app_flask/` simulates the portal of a healthcare group. On startup the server loads `modello_epilessia.pkl`, a scikit-learn `RandomForestClassifier` (200 trees, classes `[2, 3]`, 4094 input features), and serves two role-based interfaces built with Bootstrap 5 and Chart.js.

<table>
<tr>
<th width="50%">👤 Patient portal <code>/</code></th>
<th width="50%">🩺 Doctor portal <code>/doctor</code></th>
</tr>
<tr valign="top">
<td>

- Dashboard with next appointment, latest documents, quick actions and care team
- **Analyses** section with exam history
- EEG report page with the real trace, the doctor's clinical comment and a signal download button

</td>
<td>

- Dashboard with daily schedule, recent patients and traces awaiting review
- **Patient reporting**: open a record, upload the CSV trace, run the model
- Result page with the **AI assessment**, measured **inference time**, a full-page **signal inspection** mode (time axis, grid, mean, std) and a form to **sign and send the report**

</td>
</tr>
</table>

### Endpoints

| Method | Route | Description |
|:-:|---|---|
| `GET` | `/` | Patient portal home |
| `GET` | `/analisi` | Patient's list of analyses |
| `GET` | `/analisi/dettaglio` | EEG report detail |
| `GET` | `/get_signal` | Signal from `test_patient.csv` as JSON |
| `GET` | `/download_signal` | Download `test_patient.csv` |
| `POST` | `/predict` | Patient-side prediction (form field `file_csv`) |
| `GET` | `/doctor` | Doctor portal home |
| `POST` | `/doctor_predict` | Doctor-side prediction with timed inference |
| `POST` | `/save_report` | Simulated report save (`diagnosi`, `referto`) |

### Input format

The uploaded CSV must contain **one value per row and no header**: a single column of trace samples. The server reads it with pandas, transposes it and feeds it to the model, which expects **4094 samples**. `test_patient.csv` is a ready-to-use example.

If the model file is missing, the app starts in **mock mode** and returns a fixed prediction so the UI can still be explored.

## 📁 Repository structure

```
EpilepsyDetection/
├── EEG-data.xlsx                # source dataset
├── EEG_final.ipynb              # main notebook (analysis + training)
├── EEG_last.ipynb               # earlier / alternate version
└── app_flask/
    ├── app.py                   # Flask server, both portals
    ├── modello_epilessia.pkl    # trained Random Forest
    ├── test_patient.csv         # sample trace (4094 samples)
    └── templates/
        ├── index.html           # patient portal
        └── doctor.html          # doctor portal
```

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Data & analysis | pandas, NumPy, SciPy, Matplotlib, Seaborn |
| Modeling | scikit-learn, mlxtend, scikit-posthocs |
| Serving | Flask, joblib |
| Frontend | Bootstrap 5, Bootstrap Icons, Chart.js, Inter font |

## ⚠️ Notes and limitations

- The web app is a **demo**. Personal data, schedules, vital signs and metrics such as "91.4% confidence" are static placeholders, and saving a report does not persist anything.
- The trace chart in both portals always reads `test_patient.csv`, not the file just uploaded.
- The model was trained on a research dataset and **is not a diagnostic tool**. Do not use it for real clinical decisions.
