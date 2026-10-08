"""Generate a fully synthetic longitudinal outpatient dataset.

The synthetic dataset is provided only for functional demonstration of the
public repository. It does not reproduce the private study cohort,
its distributions, its patient records, or its reported model results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENDERS = ["Female", "Male"]
INSURANCE_TYPES = [
    "Social Security",
    "Health Insurance",
    "Organization",
    "Private",
    "Other",
]
SPECIALTIES = [
    "General Practice",
    "Internal Medicine",
    "Cardiology",
    "Neurology",
    "Orthopedics",
    "Dermatology",
    "ENT",
    "Gynecology",
]
DOCTORS = [f"DOC_{i:03d}" for i in range(1, 11)]
NEXT_SERVICES = ["Drug", "Laboratory", "Radiology", "ClinicalService"]
NEXT_PROBS = [0.35, 0.30, 0.20, 0.15]


def generate_service_sequence(rng: np.random.Generator, min_length: int = 2, max_length: int = 7) -> list[str]:
    """Create an episode sequence beginning with a physician visit."""
    length = int(rng.integers(min_length, max_length + 1))
    sequence = ["Visit"]
    for _ in range(length - 1):
        sequence.append(rng.choice(NEXT_SERVICES, p=NEXT_PROBS))
    return sequence


def generate_synthetic_data(
    n_patients: int = 300,
    min_episodes: int = 1,
    max_episodes: int = 5,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic episode-level longitudinal data."""
    if n_patients < 2:
        raise ValueError("n_patients must be at least 2")
    if min_episodes < 1 or max_episodes < min_episodes:
        raise ValueError("Require 1 <= min_episodes <= max_episodes")

    rng = np.random.default_rng(random_state)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")

    rows: list[dict] = []

    for patient_number in range(1, n_patients + 1):
        national_code = f"SYNTH_{patient_number:06d}"
        age = int(np.clip(rng.normal(45, 18), 18, 85))
        gender = rng.choice(GENDERS)
        insurance = rng.choice(INSURANCE_TYPES, p=[0.35, 0.25, 0.15, 0.15, 0.10])
        n_episodes = int(rng.integers(min_episodes, max_episodes + 1))

        first_date = start + pd.Timedelta(days=int(rng.integers(0, max(1, (end - start).days - 120))))
        previous_date = first_date

        for episode_number in range(1, n_episodes + 1):
            if episode_number > 1:
                gap_days = int(rng.choice([7, 14, 21, 30, 45, 60, 90], p=[0.12, 0.12, 0.14, 0.20, 0.16, 0.14, 0.12]))
                episode_start = previous_date + pd.Timedelta(days=gap_days)
            else:
                episode_start = first_date

            if episode_start > end:
                break

            specialty = rng.choice(SPECIALTIES)
            doctor = rng.choice(DOCTORS)
            sequence = generate_service_sequence(rng)
            unique_service_count = len(set(sequence))
            total_events = len(sequence)

            duration_days = int(rng.choice(
                [0, 1, 2, 3, 5, 7, 10, 14],
                p=[0.12, 0.18, 0.20, 0.18, 0.12, 0.09, 0.07, 0.04],
            ))
            episode_end = min(episode_start + pd.Timedelta(days=duration_days), end)
            duration_days = int((episode_end - episode_start).days)

            base_cost = rng.lognormal(mean=8.0, sigma=0.65)
            multiplier = 1.0
            multiplier += 0.45 if "Radiology" in sequence else 0
            multiplier += 0.20 if "Laboratory" in sequence else 0
            multiplier += 0.25 if "ClinicalService" in sequence else 0
            multiplier += 0.15 if "Drug" in sequence else 0
            total_cost = base_cost * multiplier * (1 + 0.08 * total_events)
            total_cost *= rng.lognormal(mean=0, sigma=0.15)

            rows.append({
                "NationalCode": national_code,
                "EpisodeID": f"SYN_{patient_number:06d}_E{episode_number:03d}",
                "EpisodeStartDate": episode_start,
                "EpisodeEndDate": episode_end,
                "TotalEpisodeCost": round(float(total_cost), 2),
                "UniqueServiceCount": unique_service_count,
                "TotalEventsInEpisode": total_events,
                "EpisodeDurationDays": duration_days,
                "ServiceSequence": " -> ".join(sequence),
                "PatientAge": age,
                "PatientGender": gender,
                "InsuranceType": insurance,
                "PrimarySpecialty": specialty,
                "DoctorID": doctor,
            })
            previous_date = episode_start

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No synthetic episodes were generated")

    df["EpisodeStartDate"] = pd.to_datetime(df["EpisodeStartDate"])
    df["EpisodeEndDate"] = pd.to_datetime(df["EpisodeEndDate"])
    return df.sort_values(["NationalCode", "EpisodeStartDate", "EpisodeID"]).reset_index(drop=True)


def validate_synthetic_data(df: pd.DataFrame) -> None:
    """Validate the public demo dataset structure."""
    required = [
        "NationalCode", "EpisodeID", "EpisodeStartDate", "EpisodeEndDate",
        "TotalEpisodeCost", "UniqueServiceCount", "TotalEventsInEpisode",
        "EpisodeDurationDays", "ServiceSequence", "PatientAge", "PatientGender",
        "InsuranceType", "PrimarySpecialty", "DoctorID",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["EpisodeID"].duplicated().any():
        raise ValueError("EpisodeID must be unique")
    if df["NationalCode"].nunique() < 2:
        raise ValueError("At least two patients are required")
    if (df["EpisodeDurationDays"] < 0).any() or (df["TotalEpisodeCost"] < 0).any():
        raise ValueError("Duration and cost must be non-negative")
    if not df["ServiceSequence"].astype(str).str.startswith("Visit -> ").all():
        raise ValueError("Every sequence must begin with Visit and contain another service")
    if not set(df["InsuranceType"].unique()).issubset(set(INSURANCE_TYPES)):
        raise ValueError("Unexpected insurance category")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo data")
    parser.add_argument("--n-patients", type=int, default=300)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_demo.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_data(
        n_patients=args.n_patients,
        min_episodes=args.min_episodes,
        max_episodes=args.max_episodes,
        random_state=args.random_state,
    )
    validate_synthetic_data(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("Synthetic dataset generated successfully.")
    print(f"Output: {args.output}")
    print(f"Rows/episodes: {len(df):,}")
    print(f"Patients: {df['NationalCode'].nunique():,}")
    print(f"Average episodes/patient: {len(df) / df['NationalCode'].nunique():.2f}")
    print("Insurance categories:", ", ".join(sorted(df["InsuranceType"].unique())))
    print("This data are synthetic and are not the study dataset.")


if __name__ == "__main__":
    main()
