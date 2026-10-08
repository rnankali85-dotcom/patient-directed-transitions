"""Run a lightweight end-to-end demonstration on synthetic data.

This script is intentionally separate from the private study analysis.
It uses the public synthetic dataset and a small hyperparameter search so
that a new user can verify the repository installation quickly.
"""

from pathlib import Path

import pandas as pd

from src.evaluation import run_statistical_evaluation
from src.pipeline import JAMIAFinalPipeline, TARGET_COLUMNS


def main() -> None:
    data_path = Path("data/synthetic_demo.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run: python src/generate_synthetic_data.py"
        )

    df = pd.read_csv(data_path, parse_dates=["EpisodeStartDate", "EpisodeEndDate"])

    pipeline = JAMIAFinalPipeline(df, model_dir="models_demo", random_state=42)
    pipeline.split_data()
    pipeline.apply_binary_targets()
    pipeline.build_features()

    # n_iter=2 is deliberately used for a fast smoke test.
    # The manuscript analysis used the study configuration (n_iter=30).
    results = pipeline.run_all_targets(n_iter=2)
    feature_sets = pipeline.get_feature_sets()

    stats = run_statistical_evaluation(
        results=results,
        train_df=pipeline.train_df,
        test_df=pipeline.test_df,
        feature_sets=feature_sets,
        targets=list(TARGET_COLUMNS),
        bootstrap_iterations=50,
    )

    output = Path("results/synthetic_demo_statistics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output, index=False)

    print("\nSynthetic end-to-end demonstration completed.")
    print(stats.to_string(index=False))
    print(f"\nSaved: {output}")
    print("\nNote: demo metrics are NOT the manuscript results.")


if __name__ == "__main__":
    main()
