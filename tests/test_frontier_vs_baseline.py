"""Contract tests for the Top-K frontier versus covariate baseline stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repo_expo_hoi_bag.stages import compute_frontier_vs_baseline as stage
from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier


def _wide(data: dict[str, list[float]], countries=("A", "B", "C", "D")) -> pd.DataFrame:
    frame = pd.DataFrame(data, index=list(countries)).T
    frame.index.name = "candidate_id"
    frame.columns.name = "fold_country"
    return frame


class TestAlgebraicEquivalence:
    """The premise of the whole brief: median-delta is not a new analysis."""

    def test_median_of_differences_equals_median_minus_constant(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            r2 = rng.normal(scale=0.2, size=20)
            baseline = float(rng.normal(scale=0.3))
            assert np.median(r2 - baseline) == pytest.approx(np.median(r2) - baseline)

    def test_holds_for_even_and_odd_k(self):
        for k in (5, 10, 20, 51):
            r2 = np.linspace(0.0, 1.0, k)
            baseline = 0.37
            assert np.median(r2 - baseline) == pytest.approx(np.median(r2) - baseline)

    def test_mean_is_a_genuinely_different_statistic(self):
        # A skewed frontier: the mean moves with the left tail, the median does
        # not. This is why the mean is the new estimand.
        r2 = np.array([0.5] * 19 + [-5.0])
        baseline = 0.4
        delta = r2 - baseline
        assert np.median(delta) == pytest.approx(0.1)
        assert delta.mean() < 0
        assert not np.isclose(delta.mean(), np.median(delta))


class TestFrontierSelection:
    def test_evaluation_country_never_ranks_its_own_frontier(self):
        frame = _wide({"star": [0.0, 0.0, 0.0, 9.0], "steady": [0.5, 0.5, 0.5, 0.5]})
        assert frontier.top_k(frame, 1, exclude="D") == ["steady"]
        assert frontier.top_k(frame, 1, exclude="A") == ["star"]

    def test_exactly_k_unique_ids(self):
        frame = _wide({f"c{i}": [i / 10] * 4 for i in range(30)})
        for k in (10, 20):
            selected = frontier.top_k(frame, k, exclude="B")
            assert len(selected) == len(set(selected)) == k


class TestCountryStatistics:
    def test_mean_delta_is_the_arithmetic_mean_of_candidate_deltas(self):
        r2 = np.array([0.30, 0.20, 0.10, 0.40])
        baseline = 0.25
        deltas = r2 - baseline
        assert deltas.mean() == pytest.approx((0.05 - 0.05 - 0.15 + 0.15) / 4)
        assert deltas.mean() == pytest.approx(r2.mean() - baseline)

    def test_positive_fraction_counts_strictly_positive_members(self):
        deltas = np.array([0.1, 0.0, -0.1, 0.2])
        assert float((deltas > 0).mean()) == pytest.approx(0.5)
        assert int((deltas < 0).sum()) == 1

    def test_all_positive_flag(self):
        assert bool((np.array([0.1, 0.2]) > 0).all())
        assert not bool((np.array([0.1, 0.0]) > 0).all())


class TestInference:
    def test_country_is_the_unit_not_the_candidate(self):
        # 4 countries x 20 candidates: the test must see 4 observations, so the
        # Wilcoxon n can never exceed the number of countries.
        summary = pd.DataFrame(
            {
                "bag": ["structural"] * 4,
                "arm": ["o_min"] * 4,
                "rung": ["xgb_tree_d3"] * 4,
                "K": [20] * 4,
                "country": list("ABCD"),
                "mean_frontier_delta": [0.01, 0.02, 0.015, 0.012],
                "positive_fraction": [0.9, 0.8, 1.0, 0.7],
            }
        )
        tests = stage.run_tests(summary, draws=200)
        primary = tests[tests["test"].eq("primary_mean_frontier_gain")]
        assert int(primary["n_countries"].iloc[0]) == 4

    def test_no_candidate_level_p_values_are_produced(self):
        summary = pd.DataFrame(
            {
                "bag": ["structural"] * 6,
                "arm": ["o_min"] * 6,
                "rung": ["xgb_tree_d3"] * 6,
                "K": [20] * 6,
                "country": list("ABCDEF"),
                "mean_frontier_delta": [0.01, 0.02, 0.015, 0.012, 0.008, 0.02],
                "positive_fraction": [0.9, 0.8, 1.0, 0.7, 0.6, 0.95],
            }
        )
        tests = stage.run_tests(summary, draws=200)
        assert set(tests["test"]) == {"primary_mean_frontier_gain", "secondary_positivity_vs_half"}
        assert tests["candidate_dependence_note"].str.contains("not independent").all()

    def test_zero_gain_gives_a_null_result(self):
        summary = pd.DataFrame(
            {
                "bag": ["structural"] * 6,
                "arm": ["o_min"] * 6,
                "rung": ["xgb_tree_d3"] * 6,
                "K": [20] * 6,
                "country": list("ABCDEF"),
                "mean_frontier_delta": [0.0] * 6,
                "positive_fraction": [0.5] * 6,
            }
        )
        tests = stage.run_tests(summary, draws=200)
        primary = tests[tests["test"].eq("primary_mean_frontier_gain")].iloc[0]
        assert primary["wilcoxon_p_raw"] == 1.0

    def test_test_stays_two_sided(self):
        # A strongly negative effect must produce a small P, which a one-sided
        # "greater" test would not.
        summary = pd.DataFrame(
            {
                "bag": ["structural"] * 8,
                "arm": ["o_min"] * 8,
                "rung": ["xgb_tree_d3"] * 8,
                "K": [20] * 8,
                "country": list("ABCDEFGH"),
                "mean_frontier_delta": [-0.05] * 8,
                "positive_fraction": [0.0] * 8,
            }
        )
        tests = stage.run_tests(summary, draws=200)
        primary = tests[tests["test"].eq("primary_mean_frontier_gain")].iloc[0]
        assert primary["wilcoxon_p_raw"] < 0.05
        assert primary["mean_country_frontier_delta"] < 0


class TestHolmFamilies:
    def _summary(self, k=20):
        rows = []
        for bag in ("structural", "functional"):
            for arm in ("o_min", "o_max"):
                for rung in ("xgb_tree_d2", "xgb_tree_d3"):
                    for index, country in enumerate("ABCDEF"):
                        rows.append(
                            {
                                "bag": bag,
                                "arm": arm,
                                "rung": rung,
                                "K": k,
                                "country": country,
                                "mean_frontier_delta": 0.01 + index * 0.001,
                                "positive_fraction": 0.8,
                            }
                        )
        return pd.DataFrame(rows)

    def test_primary_family_has_exactly_four_tests_per_bag(self):
        tests = stage.run_tests(self._summary(), draws=200)
        stage.validate_families(tests)
        primary = tests[tests["test"].eq("primary_mean_frontier_gain")]
        assert set(primary.groupby("multiplicity_family").size()) == {4}

    def test_families_are_separated_by_bag_and_k(self):
        tests = stage.run_tests(
            pd.concat([self._summary(10), self._summary(20)], ignore_index=True), draws=200
        )
        stage.validate_families(tests)
        primary = tests[tests["test"].eq("primary_mean_frontier_gain")]
        assert primary["multiplicity_family"].nunique() == 4  # 2 bags x 2 K values

    def test_ols_and_d1_are_not_in_the_primary_scope(self):
        assert "ols" not in stage.RUNGS
        assert "xgb_tree_d1" not in stage.RUNGS
        assert stage.RUNGS == ("xgb_tree_d2", "xgb_tree_d3")

    def test_wrong_family_size_is_rejected(self):
        frame = pd.DataFrame(
            {
                "test": ["primary_mean_frontier_gain"] * 3,
                "multiplicity_family": ["structural|primary_mean_frontier_gain|K20"] * 3,
            }
        )
        with pytest.raises(ValueError, match="expected 4"):
            stage.validate_families(frame)


class TestDeliveredArtifacts:
    ROOT = stage.ROOT / "outputs/sensitivity/frontier_vs_baseline/main_k10"

    @pytest.fixture(scope="class")
    def summary(self):
        path = self.ROOT / "frontier_vs_baseline_country_summary.csv"
        if not path.is_file():
            pytest.skip("frontier-vs-baseline has not been generated")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def audit(self):
        path = self.ROOT / "frontier_vs_baseline_candidate_audit.csv"
        if not path.is_file():
            pytest.skip("candidate audit has not been generated")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def tests_frame(self):
        path = self.ROOT / "frontier_vs_baseline_tests.csv"
        if not path.is_file():
            pytest.skip("tests have not been generated")
        return pd.read_csv(path)

    def test_primary_families_are_correct(self, tests_frame):
        stage.validate_families(tests_frame)

    def test_audit_holds_exactly_k_candidates_per_country(self, audit):
        counts = audit.groupby(["bag", "arm", "rung", "country", "K"]).size().reset_index(name="n")
        assert (counts["n"] == counts["K"]).all()

    def test_baseline_is_unique_per_bag_rung_country(self, audit):
        unique = audit.groupby(["bag", "rung", "country"])["baseline_country_r2"].nunique()
        assert (unique == 1).all()

    def test_mean_delta_reproduces_from_the_audit(self, summary, audit):
        recomputed = (
            audit.groupby(["bag", "arm", "rung", "country", "K"])["delta_r2"].mean().rename("mean").reset_index()
        )
        merged = summary.merge(recomputed, on=["bag", "arm", "rung", "country", "K"], validate="one_to_one")
        assert np.allclose(merged["mean"], merged["mean_frontier_delta"])

    def test_median_delta_equals_prior_frontier_median_minus_baseline(self, summary):
        """The stated algebra, checked against the delivered frontier output."""
        prior_path = (
            stage.ROOT / "outputs/sensitivity/performance_frontier/main_k10/frontier_country_scores.csv"
        )
        if not prior_path.is_file():
            pytest.skip("prior frontier scores are unavailable")
        prior = pd.read_csv(prior_path).rename(columns={"excluded_country": "country"})
        merged = summary.merge(
            prior[["bag", "arm", "rung", "country", "K", "frontier_median_r2"]],
            on=["bag", "arm", "rung", "country", "K"],
            validate="one_to_one",
        )
        assert len(merged) == len(summary)
        expected = merged["frontier_median_r2"] - merged["baseline_country_r2"]
        assert np.allclose(merged["median_frontier_delta"], expected)

    def test_positive_fraction_matches_the_audit(self, summary, audit):
        recomputed = (
            audit.groupby(["bag", "arm", "rung", "country", "K"])["delta_positive"].mean().rename("frac").reset_index()
        )
        merged = summary.merge(recomputed, on=["bag", "arm", "rung", "country", "K"], validate="one_to_one")
        assert np.allclose(merged["frac"], merged["positive_fraction"])

    def test_primary_k_is_twenty(self, tests_frame):
        assert stage.PRIMARY_K == 20
        assert 20 in set(tests_frame["K"])

    def test_table_reproduces_the_tests(self, tests_frame):
        source = stage.ROOT / (
            "outputs/main/paper/complete/tables/source_data/frontier_vs_baseline/"
            "frontier_vs_baseline_comparisons.csv"
        )
        if not source.is_file():
            pytest.skip("publication table has not been generated")
        table = pd.read_csv(source)
        primary = tests_frame[
            tests_frame["test"].eq("primary_mean_frontier_gain") & tests_frame["K"].eq(20)
        ]
        assert len(table) == len(primary)
        assert np.allclose(
            sorted(table["Mean frontier ΔR²"]), sorted(primary["mean_country_frontier_delta"])
        )

    def test_figure_source_data_reproduces_the_tests(self, tests_frame):
        source = stage.ROOT / (
            "outputs/main/paper/complete/figures/supplementary/source_data/"
            "fig_frontier_vs_baseline_hpo_k10_source_data_all_cells.csv"
        )
        if not source.is_file():
            pytest.skip("figure source data has not been generated")
        drawn = pd.read_csv(source)
        primary = tests_frame[
            tests_frame["test"].eq("primary_mean_frontier_gain") & tests_frame["K"].eq(20)
        ]
        merged = drawn.merge(primary, on=["bag", "arm", "rung"], validate="one_to_one")
        assert len(merged) == len(drawn)
        assert np.allclose(merged["mean_frontier_delta"], merged["mean_country_frontier_delta"])
        assert np.allclose(merged["p_holm_adjusted"], merged["holm_p"])
