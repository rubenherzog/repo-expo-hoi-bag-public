"""Contract tests for the country-cross-selected performance-frontier stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repo_expo_hoi_bag.stages import compute_performance_frontier as stage
from repo_expo_hoi_bag.stages.run_selection_stability_diagnostic import _jaccard


def _wide(data: dict[str, list[float]], countries=("A", "B", "C", "D")) -> pd.DataFrame:
    """A candidate x country R2 matrix shaped exactly like the production pivot.

    The index must be named ``candidate_id``: the selection rule resets it and
    tie-breaks on that column.
    """
    frame = pd.DataFrame(data, index=list(countries)).T
    frame.index.name = "candidate_id"
    frame.columns.name = "fold_country"
    return frame


class TestCrossSelection:
    def test_scoring_country_never_ranks_its_own_frontier(self):
        # "star" is worthless everywhere except in D, where it is spectacular.
        # A frontier for D must not select it; a frontier for A may.
        frame = _wide(
            {
                "star": [0.0, 0.0, 0.0, 10.0],
                "good1": [0.5, 0.5, 0.5, 0.5],
                "good2": [0.4, 0.4, 0.4, 0.4],
                "bad": [0.1, 0.1, 0.1, 0.1],
            }
        )
        assert "star" not in stage.top_k(frame, 2, exclude="D")
        assert "star" in stage.top_k(frame, 2, exclude="A")
        # The global frontier, which uses every country, does pick it up.
        assert "star" in stage.top_k(frame, 2)

    def test_exclusion_changes_only_the_selection_metric(self):
        # The evaluated value is always the stored R2 in the held-out country,
        # regardless of which countries were used to rank.
        frame = _wide({"a": [0.1, 0.9, 0.9, 0.9], "b": [0.2, 0.1, 0.1, 0.1]})
        selected = stage.top_k(frame, 1, exclude="A")
        assert selected == ["a"]
        median, mean = stage.frontier_score(frame, selected, "A")
        assert median == pytest.approx(0.1)
        assert mean == pytest.approx(0.1)

    def test_selects_exactly_k_unique_candidates(self):
        frame = _wide({f"c{i}": [i / 10] * 4 for i in range(8)})
        for k in (1, 3, 8):
            selected = stage.top_k(frame, k, exclude="B")
            assert len(selected) == k
            assert len(set(selected)) == k

    def test_ranking_uses_the_country_balanced_mean(self):
        # "wide" has the higher mean; "peaky" has the higher single value.
        # The production rule ranks on the unweighted mean.
        frame = _wide({"peaky": [0.9, 0.0, 0.0, 0.0], "wide": [0.3, 0.3, 0.3, 0.3]})
        assert stage.top_k(frame, 1) == ["wide"]

    def test_ties_break_on_ascending_candidate_id(self):
        frame = _wide({"zebra": [0.5] * 4, "alpha": [0.5] * 4, "mango": [0.5] * 4})
        assert stage.top_k(frame, 2) == ["alpha", "mango"]

    def test_too_few_candidates_is_an_error(self):
        frame = _wide({"a": [0.1] * 4, "b": [0.2] * 4})
        with pytest.raises(ValueError, match="Only 2 candidates"):
            stage.top_k(frame, 20)


class TestFrontierScore:
    def test_median_not_maximum(self):
        frame = _wide({"a": [0.1] * 4, "b": [0.2] * 4, "c": [9.0] * 4})
        median, mean = stage.frontier_score(frame, ["a", "b", "c"], "A")
        assert median == pytest.approx(0.2)
        assert mean == pytest.approx(3.1)

    def test_frontier_is_robust_to_one_outlying_member(self):
        base = {f"c{i}": [0.3] * 4 for i in range(5)}
        clean, _ = stage.frontier_score(_wide(base), list(base), "A")
        base["c0"] = [-99.0] * 4
        spoiled, _ = stage.frontier_score(_wide(base), list(base), "A")
        assert clean == pytest.approx(spoiled)


class TestJaccard:
    def test_candidate_id_jaccard(self):
        assert _jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == pytest.approx(0.5)

    def test_identical_sets_are_one_and_disjoint_are_zero(self):
        assert _jaccard(frozenset({"a"}), frozenset({"a"})) == pytest.approx(1.0)
        assert _jaccard(frozenset({"a"}), frozenset({"b"})) == pytest.approx(0.0)

    def test_candidate_and_exposure_jaccards_are_distinct_quantities(self):
        # Two different candidates can share every exposure: candidate-ID
        # Jaccard is 0 while exposure Jaccard is 1. Conflating them would
        # report a frontier as unstable when its exposures are identical.
        predictors = {"cand1": frozenset({"x", "y"}), "cand2": frozenset({"x", "y"})}
        assert _jaccard(frozenset({"cand1"}), frozenset({"cand2"})) == pytest.approx(0.0)
        left = frozenset().union(*(predictors[c] for c in ["cand1"]))
        right = frozenset().union(*(predictors[c] for c in ["cand2"]))
        assert _jaccard(left, right) == pytest.approx(1.0)


class TestStabilityStats:
    def test_mean_is_reported_alongside_the_distribution(self):
        stats = stage._stability_stats([0.0, 0.5, 1.0], "top20_candidate_jaccard")
        assert stats["top20_candidate_jaccard_mean"] == pytest.approx(0.5)
        assert stats["top20_candidate_jaccard_median"] == pytest.approx(0.5)
        assert stats["top20_candidate_jaccard_min"] == pytest.approx(0.0)
        assert stats["top20_candidate_jaccard_max"] == pytest.approx(1.0)

    def test_mean_and_median_differ_on_a_skewed_distribution(self):
        stats = stage._stability_stats([1.0, 1.0, 1.0, 0.0], "p")
        assert stats["p_median"] == pytest.approx(1.0)
        assert stats["p_mean"] == pytest.approx(0.75)


class TestMatchedRungUnion:
    def test_union_is_the_same_identities_scored_under_both_rungs(self):
        high = _wide({"a": [0.5] * 4, "b": [0.4] * 4, "c": [0.3] * 4})
        low = _wide({"a": [0.2] * 4, "b": [0.1] * 4, "c": [0.9] * 4})
        union = sorted(set(stage.top_k(high, 1, exclude="A")) | set(stage.top_k(low, 1, exclude="A")))
        assert union == ["a", "c"]
        # Both rungs are scored on exactly this set, so any difference is a
        # model-level effect rather than a change of frontier membership.
        assert float(np.median(high.loc[union, "A"])) == pytest.approx(0.4)
        assert float(np.median(low.loc[union, "A"])) == pytest.approx(0.55)


class TestHolmFamilies:
    def test_expected_family_sizes(self):
        assert stage.EXPECTED_FAMILY_SIZES == {
            "synergy_vs_redundancy": 4,
            "model_complexity": 6,
            "vs_covariate_baseline": 8,
            "vs_cross_selected_single": 8,
            "matched_candidate_model_complexity": 3,
        }

    def test_wrong_family_size_is_rejected(self):
        frame = pd.DataFrame(
            {"multiplicity_family": ["f"] * 5, "comparison_family": ["model_complexity"] * 5}
        )
        with pytest.raises(ValueError, match="expected 6"):
            stage.validate_families(frame)

    def test_families_are_separated_by_k(self):
        # A K=10 family must not be Holm-corrected together with K=20.
        assert "K10" in f"structural|synergy_vs_redundancy|K10"


class TestDeliveredArtifacts:
    ROOT = stage.ROOT / "outputs/sensitivity/performance_frontier/main_k10"

    @pytest.fixture(scope="class")
    def tests_frame(self):
        path = self.ROOT / "frontier_comparison_tests.csv"
        if not path.is_file():
            pytest.skip("frontier comparisons have not been generated")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def scores(self):
        path = self.ROOT / "frontier_country_scores.csv"
        if not path.is_file():
            pytest.skip("frontier scores have not been generated")
        return pd.read_csv(path)

    def test_primary_analysis_is_k20(self, tests_frame):
        assert set(tests_frame["K"]) == {20}

    def test_all_families_have_the_expected_size(self, tests_frame):
        stage.validate_families(tests_frame)

    def test_exactly_k_candidates_selected_in_every_cell(self, scores):
        counts = scores["selected_candidate_ids"].str.split("|").apply(len)
        assert (counts == scores["K"]).all()
        unique = scores["selected_candidate_ids"].str.split("|").apply(lambda ids: len(set(ids)))
        assert (unique == scores["K"]).all()

    def test_paired_countries_match_within_a_bag(self, tests_frame):
        for bag, group in tests_frame.groupby("bag"):
            assert group["n_countries"].nunique() == 1

    def test_jaccard_values_are_in_range(self, scores):
        for column in (
            "global_frontier_candidate_jaccard",
            "global_frontier_exposure_jaccard",
            "global_frontier_domain_jaccard",
        ):
            assert scores[column].between(0.0, 1.0).all()

    def test_table_values_equal_figure_values(self, tests_frame):
        source = stage.ROOT / (
            "outputs/main/paper/complete/figures/supplementary/source_data/"
            "fig_s13_performance_frontier_comparisons_hpo_k10_source_data_all_cells.csv"
        )
        if not source.is_file():
            pytest.skip("figure source data has not been generated")
        drawn = pd.read_csv(source)
        merged = drawn.merge(
            tests_frame,
            on=["bag", "comparison_family", "arm", "rung_a", "rung_b"],
            validate="one_to_one",
        )
        assert len(merged) == len(drawn)
        assert np.allclose(merged["delta_r2"], merged["delta_mean_r2"])
        assert np.allclose(merged["p_holm_adjusted"], merged["holm_p"])

    def test_publication_table_matches_the_statistics(self, tests_frame):
        source = stage.ROOT / (
            "outputs/main/paper/complete/tables/source_data/"
            "performance_frontier/performance_frontier_comparisons.csv"
        )
        if not source.is_file():
            pytest.skip("publication table has not been generated")
        table = pd.read_csv(source)
        assert len(table) == len(tests_frame)
        assert np.allclose(sorted(table["ΔR²"]), sorted(tests_frame["delta_mean_r2"]))
