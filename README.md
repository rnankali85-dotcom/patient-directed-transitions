# Predicting Episode-Level Healthcare Utilization From Historical Service Pathways

Public code package accompanying the study:

**Predicting Episode-Level Healthcare Utilization From Historical Service Pathways: A Patient-Specific Directed Transition-Based Machine Learning Approach**

## What this repository contains

This repository provides the analysis pipeline for evaluating whether
patient-specific directed transitions between outpatient healthcare service
categories add predictive information beyond demographic, care-context, and
historical utilization features.

Four episode-level outcomes are evaluated:

- `isHighCost`
- `isComplexEpisode`
- `isHighResource`
- `isLongEpisode`

Three nested model configurations are compared:

1. **Baseline** — demographic and care-context features.
2. **Aggregate** — Baseline + historical utilization aggregates.
3. **Proposed** — Aggregate + patient-specific directed transition features.

The predictive model is XGBoost. Evaluation includes AUROC, AUPRC, Brier
score, MCC, out-of-fold threshold optimization, paired bootstrap AUROC
comparisons, calibration analysis, and SHAP-based model attribution.

## Important: synthetic data vs. study data

The public repository contains `data/synthetic_demo.csv` solely for functional
demonstration. It is **not** the study dataset and must not be described as a
reproduction of the study cohort, distributions, or reported results.

The synthetic generator can be rerun with:

```bash
python src/generate_synthetic_data.py
```

The generator includes an organizational insurance category (`Organization`,
بیمه سازمانی) and repeated episodes per synthetic patient so that historical
features and directed transitions are genuinely exercised.

## Data privacy

The underlying patient-level study dataset is **not included** in this
repository. It was derived from an operational healthcare information system
and contains sensitive healthcare information.

The public repository does not contain:

- real patient identifiers;
- hashed or unhashed national identifiers from the study;
- patient-level train/test files;
- raw healthcare records;
- institution-specific database exports;
- trained models fitted to the private institutional dataset;
- patient-level predictions.

A hash is not treated as sufficient anonymization for public release. The
private dataset must remain in an appropriately authorized environment.

See [`data/README.md`](data/README.md).

## Expected input schema for authorized local data

The analysis pipeline expects an episode-level dataframe containing compatible
fields including:

- `NationalCode`
- `EpisodeID`
- `EpisodeStartDate`
- `EpisodeEndDate`
- `TotalEpisodeCost`
- `UniqueServiceCount`
- `TotalEventsInEpisode`
- `EpisodeDurationDays`
- `ServiceSequence`
- `PatientAge`
- `PatientGender`
- `InsuranceType`
- `PrimarySpecialty`
- `DoctorID`

`NationalCode` is used only as a patient grouping/history key. The public
pipeline hashes it internally and excludes it from the predictive feature
space.

## Directed transition representation

For each patient, prior episode service sequences are converted into
cumulative directed transition counts. Direction is preserved, so for example:

- `TR_Visit->Drug`
- `TR_Drug->Visit`

are different features.

The current episode is not added to its own historical transition vector.

## Patient-level split and temporal information constraint

The final train/test split uses `GroupShuffleSplit` with the patient identifier
as the grouping variable, so a patient is assigned exclusively to either the
training or held-out test partition.

For a held-out patient, the patient's own earlier observed episodes may be
used when constructing historical features for that patient's later test
episodes. The current episode and future episodes are not used to construct
its historical predictors.

Hyperparameter optimization uses conventional five-fold cross-validation
within the training partition. The study does not claim group-aware
hyperparameter tuning. Group-aware folds are used for the out-of-fold
threshold-selection stage.

## Explainability

SHAP TreeExplainer is used for model attribution. SHAP values are interpreted
as model-attribution measures and **not** as causal effects.

Four selected SHAP figures from the study are provided in `figures/`, subject
to the applicable data-governance permissions.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── src/
│   ├── __init__.py
│   ├── generate_synthetic_data.py
│   ├── pipeline.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── run_demo.py
├── notebooks/
│   └── analysis.ipynb
├── data/
│   ├── README.md
│   └── synthetic_demo.csv
├── docs/
│   └── data_dictionary.md
├── figures/
│   ├── Fig1_SHAP_Complex.png
│   ├── Fig2_SHAP_HighResource.png
│   ├── Fig3_SHAP_HighCost.png
│   └── Fig4_SHAP_LongEpisode.png
└── results/
    ├── README.md
    └── statistical_significance.csv
```

## Installation

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fast repository test

After installation:

```bash
python src/run_demo.py
```

This performs a small end-to-end smoke test on the synthetic dataset. The demo
uses only two randomized hyperparameter configurations so it finishes much
faster than the full study analysis.

The demo creates local model files under `models_demo/` and a temporary
`results/synthetic_demo_statistics.csv`; both are ignored by Git.

## Public notebook

Open:

`notebooks/analysis.ipynb`

The notebook uses the synthetic dataset by default. It is intentionally
separated from the private study data workflow.

## Study analysis configuration

The manuscript analysis used 30 randomized hyperparameter configurations per
model/target (`n_iter=30`), patient-level splitting, training-set-only Q75
threshold definition, group-aware OOF threshold selection, paired bootstrap
comparisons, and SHAP attribution. The synthetic demonstration is only a
functional test and does not reproduce those manuscript results.

## Results

`results/statistical_significance.csv` contains aggregated, non-patient-level
statistics used in the manuscript. No patient-level predictions or fitted
private-data models are published.

## Ethics and data governance

Users adapting this code are responsible for obtaining any required
institutional ethics approval, waiver, data-use authorization, privacy review,
and other permissions before processing healthcare data.

The repository itself does not grant permission to use any clinical dataset.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Once the
GitHub repository has a permanent URL, update `repository-code` in
`CITATION.cff` before publication.
