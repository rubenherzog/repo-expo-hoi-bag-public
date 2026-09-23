"""Contract tests for the landscape-level model comparison stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repo_expo_hoi_bag.stages import compute_landscape_model_comparisons as stage


def _candidate_frame(
    *,
    countries=("A", "B", "C", "D", "E", "F"),
    orders=range(3, 31),
    per_cell=20,
    r2_of=lambda objective, order, rank, country: 0.1,
) -> pd.DataFrame:
    rows = []
    for objective in ("o_min", "o_max"):
        for order in orders:
            for rank in range(per_cell):
                candidate = f"m{rank:06d}_{objective}_ord{order}_rk{rank:05d}"
                for country in countries:
                    rows.append(
                        {
                            "candidate_id": candidate,
                            "rung_id": "xgb_tree_d2",
                            "fold_country": country,
                            "n_test": 100,
                            "r2": r2_of(objective, order, rank, country),
                        }
                    )
    return pd.DataFrame(rows)


class TestLandscapeAggregation:
    def test_median_within_set_size_then_mean_over_set_sizes(self):
        # Rank 0 is an extreme outlier at every set size; the median must
        # ignore it, so the landscape score is the median of the rest.
        def r2_of(objective, order, rank, country):
            return 99.0 if rank == 0 else 0.2

        scores = stage.landscape_scores(_candidate_frame(r2_of=r2_of))
        assert set(scores["objective"]) == {"o_min", "o_max"}
        assert np.allclose(scores["landscape_r2"], 0.2)

    def test_every_set_size_contributes_equally(self):
        # One set size is much better. With 28 set sizes the landscape score
        # must move by exactly 1/28 of the difference, not be dominated by it.
        def r2_of(objective, order, rank, country):
            return 1.0 if order == 7 else 0.0

        scores = stage.landscape_scores(_candidate_frame(r2_of=r2_of))
        assert np.allclose(scores["landscape_r2"], 1 / 28)
        assert set(scores["n_orders"]) == {28}

    def test_incomplete_cell_is_an_error_not_a_smaller_median(self):
        frame = _candidate_frame()
        trimmed = frame[
            ~(
                frame["candidate_id"].str.contains("_o_min_ord5_")
                & frame["candidate_id"].str.endswith("rk00019")
            )
        ]
        with pytest.raises(ValueError, match="Incomplete candidate cells"):
            stage.landscape_scores(trimmed)

    def test_ineligible_rows_are_dropped(self):
        frame = _candidate_frame()
        frame.loc[frame["fold_country"].eq("A"), "n_test"] = 0
        scores = stage.landscape_scores(frame)
        assert "A" not in set(scores["fold_country"])

    def test_no_candidate_is_selected_on_performance(self):
        # Permuting which candidate holds which score within a set size cannot
        # change a median-based landscape: the statistic is selection-free.
        rng = np.random.default_rng(0)
        # One fixed permutation of the 20 ranks, reused for every cell, so the
        # multiset of scores per set size is identical and only the mapping
        # candidate -> score changes.
        permutation = rng.permutation(20)

        def ranked(objective, order, rank, country):
            return float(rank) / 100

        def shuffled(objective, order, rank, country):
            return float(permutation[rank]) / 100

        a = stage.landscape_scores(_candidate_frame(r2_of=ranked))
        b = stage.landscape_scores(_candidate_frame(r2_of=shuffled))
        merged = a.merge(b, on=["fold_country", "objective"], suffixes=("_a", "_b"))
        assert np.allclose(merged["landscape_r2_a"], merged["landscape_r2_b"])

    def test_native_order_range_is_honoured(self):
        # A sensitivity whose pool stops at order 10 uses its own range.
        frame = _candidate_frame(orders=range(3, 11))
        scores = stage.landscape_scores(frame, order_min=3, order_max=10)
        assert set(scores["n_orders"]) == {8}


class TestSingleAndBaseline:
    def test_single_landscape_is_a_median_not_a_maximum(self):
        frame = pd.DataFrame(
            {
                "candidate_id": [f"single_{i}" for i in range(5)] * 2,
                "fold_country": ["A"] * 5 + ["B"] * 5,
                "n_test": 100,
                "r2": [0.0, 0.1, 0.2, 0.3, 9.0] * 2,
            }
        )
        result = stage.single_exposure_landscape(frame)
        assert np.allclose(result["single_landscape_r2"], 0.2)
        assert set(result["n_single_candidates"]) == {5}

    def test_baseline_must_be_unique(self):
        frame = pd.DataFrame(
            {
                "candidate_id": ["baseline", "other"],
                "fold_country": ["A", "B"],
                "n_test": 100,
                "r2": [0.1, 0.2],
            }
        )
        with pytest.raises(ValueError, match="not unique"):
            stage.covariate_baseline(frame)

    def test_baseline_has_one_row_per_country(self):
        frame = pd.DataFrame(
            {
                "candidate_id": ["baseline"] * 3,
                "fold_country": ["A", "A", "B"],
                "n_test": 100,
                "r2": [0.1, 0.15, 0.2],
            }
        )
        with pytest.raises(ValueError, match="more than one row per country"):
            stage.covariate_baseline(frame)


class TestPairedCountryInference:
    def test_only_shared_countries_are_paired(self):
        a = pd.Series({"A": 0.2, "B": 0.3, "C": 0.4})
        b = pd.Series({"B": 0.1, "C": 0.2, "D": 0.9})
        result = stage.paired_country_test(a, b, draws=200, seed=1)
        assert result["n_countries"] == 2
        assert result["delta_mean_r2"] == pytest.approx(0.2)

    def test_country_is_the_statistical_unit(self):
        # Duplicating participants cannot change a country-level test, so the
        # result depends only on the per-country scores.
        a = pd.Series({c: 0.3 for c in "ABCDEFGH"})
        b = pd.Series({c: 0.1 for c in "ABCDEFGH"})
        result = stage.paired_country_test(a, b, draws=500, seed=2)
        assert result["n_countries"] == 8
        assert result["rank_biserial"] == pytest.approx(1.0)
        assert result["wilcoxon_p_raw"] < 0.05

    def test_identical_inputs_give_no_effect(self):
        a = pd.Series({c: 0.25 for c in "ABCDEF"})
        result = stage.paired_country_test(a, a.copy(), draws=200, seed=3)
        assert result["delta_mean_r2"] == pytest.approx(0.0)
        assert result["rank_biserial"] == pytest.approx(0.0)
        assert result["wilcoxon_p_raw"] == 1.0

    def test_rank_biserial_sign_follows_the_difference(self):
        a = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4, "E": 0.5, "F": 0.6})
        b = a + 0.05
        assert stage.paired_country_test(a, b, draws=200, seed=4)["rank_biserial"] < 0
        assert stage.paired_country_test(b, a, draws=200, seed=4)["rank_biserial"] > 0


class TestHolmFamilies:
    def test_expected_family_sizes(self):
        assert stage.EXPECTED_FAMILY_SIZES == {
            "synergy_vs_redundancy": 4,
            "model_complexity": 6,
            "vs_single_landscape": 8,
            "vs_covariate_baseline": 8,
        }

    def test_holm_is_applied_within_and_not_across_families(self):
        frame = pd.DataFrame(
            {
                "multiplicity_family": ["x"] * 4 + ["y"] * 4,
                "wilcoxon_p_raw": [0.01, 0.02, 0.03, 0.04] * 2,
            }
        )
        adjusted = stage.apply_holm(frame)
        # Holm within a family of 4 multiplies the smallest p by 4, not by 8.
        assert adjusted.loc[0, "holm_p"] == pytest.approx(0.04)
        assert adjusted.loc[4, "holm_p"] == pytest.approx(0.04)

    def test_wrong_family_size_is_rejected(self):
        frame = pd.DataFrame(
            {
                "multiplicity_family": ["f"] * 3,
                "comparison_family": ["synergy_vs_redundancy"] * 3,
            }
        )
        with pytest.raises(ValueError, match="expected 4"):
            stage.validate_families(frame)


class TestDeliveredArtifacts:
    """The delivered table, figure source data and workbook must agree."""

    ROOT = stage.ROOT / "outputs/sensitivity/landscape_comparisons/main_k10"

    @pytest.fixture(scope="class")
    def tests_frame(self):
        path = self.ROOT / "landscape_comparison_tests.csv"
        if not path.is_file():
            pytest.skip("landscape comparisons have not been generated")
        return pd.read_csv(path)

    def test_all_families_have_the_expected_size(self, tests_frame):
        stage.validate_families(tests_frame)

    def test_every_bag_and_family_is_present(self, tests_frame):
        assert set(tests_frame["bag"]) == set(stage.BAGS)
        assert len(tests_frame) == 2 * (4 + 2 * 6 + 8 + 8)

    def test_country_counts_are_consistent_within_a_bag(self, tests_frame):
        for bag, group in tests_frame.groupby("bag"):
            assert group["n_countries"].nunique() == 1

    def test_source_data_reproduces_the_statistical_table(self, tests_frame):
        source = stage.ROOT / (
            "outputs/main/paper/complete/figures/supplementary/source_data/"
            "fig_s13_landscape_model_comparisons_hpo_k10_source_data_all_cells.csv"
        )
        if not source.is_file():
            pytest.skip("figure source data has not been generated")
        drawn = pd.read_csv(source)
        merged = drawn.merge(
            tests_frame,
            on=["bag", "comparison_family", "objective", "rung_a", "rung_b"],
            validate="one_to_one",
        )
        assert len(merged) == len(drawn)
        assert np.allclose(merged["delta_r2"], merged["delta_mean_r2"])
        assert np.allclose(merged["p_holm_adjusted"], merged["holm_p"])

    def test_publication_table_reproduces_the_statistical_table(self, tests_frame):
        source = stage.ROOT / (
            "outputs/main/paper/complete/tables/source_data/"
            "landscape_comparisons/landscape_model_comparisons.csv"
        )
        if not source.is_file():
            pytest.skip("publication table has not been generated")
        table = pd.read_csv(source)
        assert len(table) == len(tests_frame)
        assert np.allclose(sorted(table["ΔR²"]), sorted(tests_frame["delta_mean_r2"]))
