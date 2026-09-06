# EEG Epilepsy Classification

Machine learning analysis of EEG recordings to distinguish epileptic patients (seizure-free intervals, on vs. off the epileptogenic zone) from healthy controls, based on the classic Andrzejak epileptic EEG dataset.

## Dataset

`EEG-data.xlsx` contains EEG recordings from 500 patients. Each recording is a single-channel EEG segment of 23.6 s, sampled into 4096 points (columns `X1`...`X4096`, sampling rate ≈ 173.61 Hz). Patients are labeled `y` from 1 to 5:

| Level | Description |
|---|---|
| 1 | Epileptic patient, seizure activity recorded on the epileptogenic zone |
| 2 | Epileptic patient, seizure-free interval recorded on the epileptogenic zone |
| 3 | Epileptic patient, seizure-free interval recorded on the hemisphere opposite the epileptogenic zone |
| 4 | Healthy subject, scalp EEG, eyes open |
| 5 | Healthy subject, scalp EEG, eyes closed |

The analysis focuses on **Level 2 vs. Level 3**, the most clinically relevant distinction (localizing seizure-free abnormal activity within the same epileptic patient).

## Notebook

`EEG_final.ipynb` walks through the full pipeline:

1. **Dataset exploration** — loading the data and plotting sample signals for each level.
2. **Outlier analysis** — standardizing signals per class and flagging patients whose recordings contain an excessive share of extreme values (> 2σ), producing both a raw dataset and an outlier-cleaned version for comparison.
3. **Model selection & GridSearchCV** — baseline Logistic Regression, then SVM (linear/poly/RBF kernels) and Random Forest, tuned via `GridSearchCV` with stratified k-fold cross-validation, evaluated with ROC curves, confusion matrices, and classification reports, on both the raw and outlier-cleaned datasets.
4. **Statistical model comparison** — 5x2cv paired t-test, majority-vote and soft-voting ensembles, AdaBoost/stacking, and a Friedman + Nemenyi test with critical-distance plots to rank all models statistically.
5. **Reduced time-window analysis** — testing classification performance using only half of the recording (11.8 s instead of 23.6 s), assessing the clinical feasibility of shorter EEG acquisitions.
6. **Frequency-domain analysis** — FFT-based preprocessing and Welch power spectral density estimation across canonical EEG frequency bands (Delta, Theta, Alpha, ...) to explore whether spectral features better separate Level 2 and Level 3 patients.

## Key findings

- SVM (RBF/poly kernels) performs slightly better on the raw dataset (with outliers), while Random Forest benefits more from outlier removal — extreme peaks are clinically meaningful (epileptic patterns) rather than noise, and SVM can exploit them as support vectors, whereas Random Forest tends to overfit to them.
- Classification performance remains nearly unchanged (accuracy ≈ 0.886, AUC ≈ 0.982) when using only half the recording length, suggesting shorter EEG acquisitions could be clinically viable.
- Statistical testing (5x2cv, Friedman/Nemenyi) is used throughout to confirm whether differences between models are significant rather than due to chance.

## Repository contents

- `EEG_final.ipynb` — main, cleaned-up analysis notebook (this is the one described above).
- `EEG_last.ipynb` — earlier/alternate version of the analysis.
- `EEG-data.xlsx` — source dataset.
- `paziente_test.csv` — sample patient data for testing.

## Requirements

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `scipy`
- `scikit-posthocs`
- `mlxtend`

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy scikit-posthocs mlxtend
```

## Usage

Open `EEG_final.ipynb` in Jupyter and run the cells sequentially; `EEG-data.xlsx` must be in the same directory.
