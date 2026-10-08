"""Core patient-specific directed-transition prediction pipeline.

This module is adapted from the study's original analysis notebook.
It intentionally does not contain or load any patient-level data.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from xgboost import XGBClassifier


TARGET_COLUMNS = {
    "isHighCost": "TotalEpisodeCost",
    "isComplexEpisode": "UniqueServiceCount",
    "isHighResource": "TotalEventsInEpisode",
    "isLongEpisode": "EpisodeDurationDays",
}

CATEGORICAL_COLUMNS = [
    "PatientGender",
    "InsuranceType",
    "PrimarySpecialty",
]

IDENTIFIER_COLUMNS = [
    "NationalCode",
    "EpisodeID",
    "DoctorID",
    "EpisodeStartDate",
    "EpisodeEndDate",
    "ServiceSequence",
    "ROW_ID",
]

DEFAULT_PARAM_DISTRIBUTION = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


class JAMIAFinalPipeline:
    """End-to-end local analysis pipeline for the study."""

    def __init__(
        self,
        df: pd.DataFrame,
        model_dir: Optional[str | Path] = None,
        random_state: int = 42,
    ) -> None:
        required = {
            "NationalCode",
            "EpisodeStartDate",
            "TotalEpisodeCost",
            "UniqueServiceCount",
            "TotalEventsInEpisode",
            "EpisodeDurationDays",
            "ServiceSequence",
            "PatientAge",
            "PatientGender",
            "InsuranceType",
            "PrimarySpecialty",
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.random_state = random_state
        self.model_dir = Path(model_dir or "models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        work = df.copy()
        # The public repository never receives the underlying data.
        # Hashing is retained from the original local analysis pipeline.
        work["NationalCode"] = work["NationalCode"].map(
            lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
        )
        work["EpisodeStartDate"] = pd.to_datetime(work["EpisodeStartDate"])
        work = work.sort_values(
            ["NationalCode", "EpisodeStartDate"]
        ).reset_index(drop=True)

        self.df = work
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.results: Dict[str, dict] = {}
        self.encoder: Optional[OneHotEncoder] = None

    def split_data(self, test_size: float = 0.20) -> None:
        """Create the patient-level 80/20 held-out split."""
        groups = self.df["NationalCode"]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self.random_state,
        )
        train_idx, test_idx = next(
            splitter.split(self.df, groups=groups)
        )
        self.train_df = self.df.iloc[train_idx].copy()
        self.test_df = self.df.iloc[test_idx].copy()

    def apply_binary_targets(self) -> None:
        """Define binary outcomes using training-set Q75 thresholds only."""
        self._require_split()

        for target, source in TARGET_COLUMNS.items():
            threshold = self.train_df[source].quantile(0.75)
            if threshold == 0:
                train_target = self.train_df[source] > 0
                test_target = self.test_df[source] > 0
            else:
                train_target = self.train_df[source] >= threshold
                test_target = self.test_df[source] >= threshold

            self.train_df[target] = train_target.astype(int)
            self.test_df[target] = test_target.astype(int)

    @staticmethod
    def _build_transition_history(group: pd.DataFrame) -> pd.DataFrame:
        """Build cumulative directed transitions using only prior episodes."""
        history: Counter = Counter()
        vectors = []

        for sequence in group["ServiceSequence"]:
            vectors.append(history.copy())

            if pd.isna(sequence):
                continue

            services = str(sequence).split(" -> ")
            for source, destination in zip(services, services[1:]):
                history[f"{source}->{destination}"] += 1

        return pd.DataFrame(vectors, index=group.index)

    def build_behavioral_transition_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Append patient-specific cumulative directed transition features."""
        transition_df = (
            df.groupby("NationalCode", group_keys=False)
            .apply(self._build_transition_history)
            .fillna(0)
        )
        transition_df.columns = [
            f"TR_{column}" for column in transition_df.columns
        ]
        return pd.concat([df, transition_df], axis=1)

    def build_features(self) -> None:
        """Create historical aggregates and train-fitted categorical encoding."""
        self._require_split()

        full_df = self.build_behavioral_transition_features(self.df.copy())
        grouped = full_df.groupby("NationalCode", sort=False)

        full_df["is_first_visit"] = grouped.cumcount() == 0
        full_df["avg_prev_cost"] = grouped["TotalEpisodeCost"].transform(
            lambda series: series.shift(1).expanding().mean()
        )
        full_df["prev_episode_count"] = grouped.cumcount()
        full_df["ROW_ID"] = np.arange(len(full_df))

        self.train_df["ROW_ID"] = self.train_df.index
        self.test_df["ROW_ID"] = self.test_df.index

        transition_cols = [
            column
            for column in full_df.columns
            if column.startswith("TR_")
        ]
        history_cols = [
            "ROW_ID",
            "is_first_visit",
            "avg_prev_cost",
            "prev_episode_count",
            *transition_cols,
        ]

        self.train_df = self.train_df.merge(
            full_df[history_cols], on="ROW_ID", how="left"
        )
        self.test_df = self.test_df.merge(
            full_df[history_cols], on="ROW_ID", how="left"
        )

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.encoder.fit(self.train_df[CATEGORICAL_COLUMNS])

        train_encoded = pd.DataFrame(
            self.encoder.transform(self.train_df[CATEGORICAL_COLUMNS]),
            columns=self.encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
            index=self.train_df.index,
        )
        test_encoded = pd.DataFrame(
            self.encoder.transform(self.test_df[CATEGORICAL_COLUMNS]),
            columns=self.encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
            index=self.test_df.index,
        )

        self.train_df = pd.concat(
            [
                self.train_df.drop(columns=CATEGORICAL_COLUMNS),
                train_encoded,
            ],
            axis=1,
        )
        self.test_df = pd.concat(
            [
                self.test_df.drop(columns=CATEGORICAL_COLUMNS),
                test_encoded,
            ],
            axis=1,
        )

    def get_feature_sets(self) -> Dict[str, list[str]]:
        """Return the three nested feature configurations."""
        self._require_features()

        transition_cols = [
            c for c in self.train_df.columns if c.startswith("TR_")
        ]
        demographic_features = [
            c
            for c in self.train_df.columns
            if c.startswith(
                ("PatientGender_", "InsuranceType_", "PrimarySpecialty_")
            )
        ]

        baseline = [
            c for c in ["PatientAge", *demographic_features]
            if c in self.train_df.columns
        ]
        aggregate = [
            c
            for c in [
                "PatientAge",
                *demographic_features,
                "avg_prev_cost",
                "prev_episode_count",
                "is_first_visit",
            ]
            if c in self.train_df.columns
        ]
        proposed = [
            c
            for c in [
                "PatientAge",
                *demographic_features,
                "avg_prev_cost",
                "prev_episode_count",
                "is_first_visit",
                *transition_cols,
            ]
            if c in self.train_df.columns
        ]

        return {
            "Baseline": baseline,
            "Aggregate": aggregate,
            "Proposed": proposed,
        }

    def fit_target(
        self,
        target_col: str,
        feature_sets: Mapping[str, Sequence[str]],
        n_iter: int = 30,
    ) -> dict:
        """Train the three nested models for one target."""
        self._require_features()

        if target_col not in TARGET_COLUMNS:
            raise ValueError(f"Unknown target: {target_col}")

        y_train = self.train_df[target_col]
        y_test = self.test_df[target_col]

        results = {}

        for model_name, features in feature_sets.items():
            model = XGBClassifier(
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=1,
            )

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=DEFAULT_PARAM_DISTRIBUTION,
                n_iter=n_iter,
                cv=5,
                random_state=self.random_state,
                scoring="roc_auc",
                n_jobs=-1,
            )

            search.fit(self.train_df[list(features)], y_train)
            best_model = search.best_estimator_

            model_path = (
                self.model_dir
                / f"model_{target_col}_{model_name}.joblib"
            )
            joblib.dump(best_model, model_path, compress=3)

            test_prob = best_model.predict_proba(
                self.test_df[list(features)]
            )[:, 1]
            test_pred = (test_prob >= 0.5).astype(int)

            results[f"{target_col}_{model_name}"] = {
                "model_path": str(model_path),
                "best_params": search.best_params_,
                "feature_count": len(features),
                "feature_names": list(features),
                "AUROC": float(roc_auc_score(y_test, test_prob)),
                "AUPRC": float(average_precision_score(y_test, test_prob)),
                "MCC": float(matthews_corrcoef(y_test, test_pred)),
                "Brier": float(brier_score_loss(y_test, test_prob)),
                "prob_pred": test_prob,
                "prob_true": y_test.to_numpy(),
            }

        self.results.update(results)
        return results

    def run_all_targets(self, n_iter: int = 30) -> Dict[str, dict]:
        """Train all targets using the nested ablation design."""
        self._require_features()
        feature_sets = self.get_feature_sets()

        for target in TARGET_COLUMNS:
            self.fit_target(target, feature_sets, n_iter=n_iter)

        return self.results

    def get_xy(
        self,
        target_col: str,
        features: Sequence[str],
    ):
        """Return train/test matrices for an explicit feature set."""
        self._require_features()
        return (
            self.train_df[list(features)].copy(),
            self.test_df[list(features)].copy(),
            self.train_df[target_col].copy(),
            self.test_df[target_col].copy(),
            self.train_df["NationalCode"].copy(),
        )

    def _require_split(self) -> None:
        if self.train_df is None or self.test_df is None:
            raise RuntimeError("Call split_data() before this operation.")

    def _require_features(self) -> None:
        self._require_split()
        if "avg_prev_cost" not in self.train_df.columns:
            raise RuntimeError(
                "Call apply_binary_targets() and build_features() first."
            )
