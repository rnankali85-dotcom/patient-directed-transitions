"""SHAP-based model attribution utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import shap


DEFAULT_TRANSLATION_MAP = {
    "عموم": "General Practitioner",
    "چشم": "Ophthalmology",
    "قلب و عروق": "Cardiology",
    "اورولوژ": "Urology",
    "روماتولوژ": "Rheumatology",
    "اعصاب و روان": "Psychiatry",
    "زنان": "OB-GYN",
    "گوش": "ENT",
    "پوست": "Dermatology",
    "ارتوپد": "Orthopedics",
    "کودکان": "Pediatrics",
    "مغز": "Neurosurgery",
    "متخصص": "Specialist",
    "گوارش و کبد": "Gastroenterology and Hepatology",
    "اجتماع": "Social Security Ins",
    "مسلح": "Organization Insurance",
    "سازمانی": "Organization Insurance",
    "آزاد": "Private/Free Ins",
}


def translate_feature_name(name: str, translation_map: Mapping[str, str] | None = None) -> str:
    """Translate selected Persian feature labels for publication figures."""
    mapping = translation_map or DEFAULT_TRANSLATION_MAP
    for source, destination in mapping.items():
        if source in name:
            return destination
    return name.replace("PrimarySpecialty_", "").replace("InsuranceType_", "Ins: ")


def generate_shap_figure(
    target_col: str,
    feature_sets: Mapping[str, Sequence[str]],
    results: Mapping[str, dict],
    test_df,
    output_path: str | Path,
    sample_size: int = 100,
    model_types: Sequence[str] = ("Proposed", "Aggregate"),
    max_display: int = 15,
) -> None:
    """Generate a two-panel SHAP summary for the selected models."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(28, 15), gridspec_kw={"wspace": 0.5})

    for axis, model_type in zip(axes, model_types):
        key = f"{target_col}_{model_type}"
        if key not in results:
            axis.axis("off")
            continue

        model = joblib.load(results[key]["model_path"])
        features = list(feature_sets[model_type])
        sample = test_df[features].sample(n=min(sample_size, len(test_df)), random_state=42).copy()
        labels = [translate_feature_name(column) for column in sample.columns]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.sca(axis)
        shap.summary_plot(
            shap_values,
            sample.values,
            feature_names=labels,
            show=False,
            plot_size=None,
            max_display=max_display,
        )
        axis.set_title(f"Model: {model_type} | Target: {target_col}", fontsize=22, pad=30)
        axis.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
