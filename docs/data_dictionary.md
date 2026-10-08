# Data Dictionary

This document describes the variables expected by the analysis pipeline.

## Patient and episode identifiers

| Variable | Description | Public repository |
|---|---|---|
| `NationalCode` | Patient-level grouping identifier available only in the authorized local dataset. It is hashed inside the pipeline for analysis. | Not included |
| `EpisodeID` | Episode identifier. | Not included |
| `DoctorID` | Physician identifier used only for local data organization/clinical context. | Not included |
| `EpisodeStartDate` | Start date of the episode. | Not included |
| `EpisodeEndDate` | End date of the episode. | Not included |

## Patient/context variables

| Variable | Description |
|---|---|
| `PatientAge` | Patient age at episode initiation. |
| `PatientGender` | Patient gender category. |
| `InsuranceType` | Insurance category at episode initiation. In the synthetic demo, categories include Social Security, Health Insurance, Organization (organizational insurance), Private, and Other. |
| `PrimarySpecialty` | Primary physician specialty/context. |

## Episode outcome variables

| Variable | Description |
|---|---|
| `TotalEpisodeCost` | Total cost associated with the episode. |
| `UniqueServiceCount` | Number of unique service categories in the episode. |
| `TotalEventsInEpisode` | Total healthcare events recorded in the episode. |
| `EpisodeDurationDays` | Episode duration in days. |
| `ServiceSequence` | Ordered sequence of service categories used to construct directed transitions. |

## Derived target variables

Targets are defined using the 75th percentile calculated on the training partition and then applied unchanged to the test partition.

- `isHighCost`: `TotalEpisodeCost` at or above the training-set Q75.
- `isComplexEpisode`: `UniqueServiceCount` at or above the training-set Q75.
- `isHighResource`: `TotalEventsInEpisode` at or above the training-set Q75.
- `isLongEpisode`: `EpisodeDurationDays` at or above the training-set Q75.

When the training-set Q75 for an outcome is zero, the positive class is defined as values greater than zero.

## Historical aggregate features

- `is_first_visit`: whether the episode is the patient's first observed episode.
- `avg_prev_cost`: mean cost of prior observed episodes available before the current episode.
- `prev_episode_count`: number of prior observed episodes.

## Directed transition features

Features beginning with `TR_` represent cumulative counts of ordered transitions observed in the patient's previous episode histories.

For example:

- `TR_Visit->Drug`
- `TR_Drug->Visit`

are separate features.

The current episode does not contribute to its own historical transition vector.
