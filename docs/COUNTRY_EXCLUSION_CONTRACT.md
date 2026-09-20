# Country-exclusion contract

Country exclusions have two intentionally independent phases. They must never be merged, inferred from one another, or silently replaced by a fallback list.

## Canonical local configuration

The sole repository-wide source is `config/country_exclusions.yaml`.

- `greedy.countries_to_remove` applies only while constructing the exposome matrix used for greedy O-information discovery.
- `bag.variants.<variant>.exclude_countries` and `exclude_diagnosis` apply only to subject-level BAG modelling and LOCO evaluation.

For the main variant `a`, BAG/LOCO excludes Egypt, Greece, and Poland, and excludes diagnoses Other, AFM, and MCI. The greedy list is currently empty. This difference is intentional: greedy discovery and subject-level prediction answer different parts of the analysis.

`bag.variants.historical_a` preserves the earlier published BAG/LOCO population
(France, Italy, Egypt, Greece, and Poland excluded; the same three diagnosis
exclusions). It is an explicit reproducibility variant for controlled historical
comparisons, not a fallback for the current main variant `a`.

## Requirements for stages

- A new or refactored greedy stage reads only the greedy exclusion list for its discovery matrix.
- A new or refactored BAG/LOCO stage reads only the selected BAG variant's country and diagnosis exclusions for its model population.
- If a compatibility stage needs environment variables, resolve them from this versioned policy at the compatibility boundary and include the resolved policy in its manifest/checkpoint signature.
- Tuning, OOF regeneration, sensitivity, residual, and candidate-evaluation stages must not fall back to an unrelated historical country list when invoked directly.
- Tests must assert the resolved countries/diagnoses for both phases and verify that the wrong phase cannot leak into the other.
