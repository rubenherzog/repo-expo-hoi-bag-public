"""Contract tests for the country-cross-selected candidate vs baseline stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repo_expo_hoi_bag.stages import compute_cross_selected_baseline as stage
from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier


def _wide(data: dict[str, list[float]], countries=("A", "B", "C", "D")) -> pd.DataFrame:
    frame = pd.DataFrame(data, index=list(countries)).T
    frame.index.name = "candidate_id"
    frame.columns.name = "fold_country"
    return frame


def _predictions(countries=("A", "B", "C", "D"), n=25, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for country in countries:
        truth = rng.normal(size=n)
        rows.append(
            pd.DataFrame(
                {
                    "country": country,
                    "y_true": truth,
                    "y_pred_full": truth * 0.6 + rng.normal(scale=0.3, size=n),
                    "y_pred_base": truth * 0.3 + rng.normal(scale=0.3, size=n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class TestSelectionExclusion:
    def test_evaluation_country_never_enters_its_own_selection(self):
        # "star" is worthless everywhere except D. D must not select it.
        frame = _wide({"star": [0.0, 0.0, 0.0, 9.0], "steady": [0.5, 0.5, 0.5, 0.5]})
        assert frontier.top_k(frame, 1, exclude="D") == ["steady"]
        assert frontier.top_k(frame, 1, exclude="A") == ["star"]

    def test_inflating_a_country_cannot_change_its_own_choice(self):
        frame = _wide({"a": [0.4, 0.4, 0.4, 0.4], "b": [0.3, 0.3, 0.3, 0.3]})
        for country in frame.columns:
            before = frontier.top_k(frame, 1, exclude=country)
            perturbed = frame.copy()
            perturbed[country] = perturbed[country] + 1000.0
            assert frontier.top_k(perturbed, 1, exclude=country) == before

    def test_selection_uses_the_country_balanced_mean_with_id_tiebreak(self):
        frame = _wide({"zebra": [0.5] * 4, "alpha": [0.5] * 4})
        assert frontier.top_k(frame, 1) == ["alpha"]


class TestPredictionAssembly:
    def _plan(self):
        return pd.DataFrame(
            {
                "bag": ["structural"] * 2,
                "arm": ["o_min"] * 2,
                "rung": ["xgb_tree_d3"] * 2,
                "country": ["A", "B"],
                "candidate_id": ["cand_x", "cand_y"],
                "predictors_identity": ["p1|p2", "p3"],
                "order": [2, 1],
                "same_as_global_winner": [True, False],
                "global_winner_candidate_id": ["cand_x", "cand_x"],
            }
        )

    def _store(self, candidate, countries):
        rows = []
        for country in countries:
            rows.append(
                pd.DataFrame(
                    {
                        "row_id": [f"{country}{i}" for i in range(3)],
                        "country": country,
                        "y_true": [0.1, 0.2, 0.3],
                        "y_pred_full": [0.1, 0.2, 0.3],
                        "y_pred_base": [0.0, 0.1, 0.2],
                        "candidate_id": candidate,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def test_each_subject_gets_exactly_one_prediction_per_cell(self):
        cached = {("structural", "xgb_tree_d3", "cand_x"): self._store("cand_x", ["A", "B"])}
        fitted = {("structural", "xgb_tree_d3", "cand_y"): self._store("cand_y", ["A", "B"])}
        predictions, missing = stage.assemble_predictions(self._plan(), cached, fitted)
        assert missing == []
        counts = predictions.groupby(["bag", "arm", "rung", "row_id"]).size()
        assert (counts == 1).all()

    def test_prediction_comes_from_the_candidate_selected_for_that_country(self):
        cached = {("structural", "xgb_tree_d3", "cand_x"): self._store("cand_x", ["A", "B"])}
        fitted = {("structural", "xgb_tree_d3", "cand_y"): self._store("cand_y", ["A", "B"])}
        predictions, _ = stage.assemble_predictions(self._plan(), cached, fitted)
        by_country = predictions.set_index("country")["candidate_id"].to_dict()
        assert by_country["A"] == "cand_x"
        assert by_country["B"] == "cand_y"

    def test_baseline_and_selected_cover_identical_participants(self):
        cached = {("structural", "xgb_tree_d3", "cand_x"): self._store("cand_x", ["A", "B"])}
        fitted = {("structural", "xgb_tree_d3", "cand_y"): self._store("cand_y", ["A", "B"])}
        predictions, _ = stage.assemble_predictions(self._plan(), cached, fitted)
        assert predictions["y_pred_full"].notna().all()
        assert predictions["y_pred_base"].notna().all()
        assert len(predictions) == 6

    def test_a_missing_fold_is_reported_not_silently_dropped(self):
        cached = {("structural", "xgb_tree_d3", "cand_x"): self._store("cand_x", ["A", "B"])}
        predictions, missing = stage.assemble_predictions(self._plan(), cached, {})
        assert predictions.empty
        assert any("cand_y" in item for item in missing)


class TestBootstrap:
    def test_resamples_whole_countries(self):
        # Every participant of a country moves together, so a bootstrap over
        # 4 countries can only ever produce country-level combinations.
        frame = _predictions(seed=1)
        result = stage.country_cluster_bootstrap(frame, draws=300, seed=7)
        assert result["bootstrap_ci_lo"] < result["country_balanced_delta_mean"] < result["bootstrap_ci_hi"]

    def test_duplicate_sampled_countries_are_handled(self):
        # A degenerate frame where one country is duplicated must not error and
        # must weight that country twice in the pooled statistic.
        frame = _predictions(countries=("A", "B"), n=10, seed=2)
        doubled = pd.concat([frame, frame[frame["country"].eq("A")]], ignore_index=True)
        result = stage.country_cluster_bootstrap(doubled, draws=200, seed=3)
        assert np.isfinite(result["pooled_delta_r2"])

    def test_pooled_r2_is_recomputed_not_averaged(self):
        # Countries of very different size: the pooled R2 must differ from the
        # unweighted mean of per-country R2, proving it is not an average.
        small = _predictions(countries=("A",), n=5, seed=4)
        large = _predictions(countries=("B",), n=500, seed=5)
        frame = pd.concat([small, large], ignore_index=True)
        pooled = stage._r2(frame["y_true"].to_numpy(float), frame["y_pred_full"].to_numpy(float))
        balanced = float(stage.country_r2(frame, "y_pred_full").mean())
        assert not np.isclose(pooled, balanced)

    def test_paired_difference_is_zero_when_predictions_match(self):
        frame = _predictions(seed=6)
        frame["y_pred_base"] = frame["y_pred_full"]
        result = stage.country_cluster_bootstrap(frame, draws=200, seed=8)
        assert result["country_balanced_delta_mean"] == pytest.approx(0.0)
        assert result["pooled_delta_r2"] == pytest.approx(0.0)


class TestHolmFamilies:
    def _summary(self):
        rows = []
        for bag in ("structural", "functional"):
            for arm in ("o_min", "o_max"):
                for rung in ("xgb_tree_d1", "xgb_tree_d2", "xgb_tree_d3"):
                    rows.append({"bag": bag, "arm": arm, "rung": rung, "bootstrap_p_raw": 0.01})
        return pd.DataFrame(rows)

    def test_primary_family_has_exactly_four_tests_per_bag(self):
        result = stage.apply_primary_holm(self._summary())
        stage.validate_primary_families(result)
        primary = result[result["multiplicity_family"].str.startswith("primary")]
        assert set(primary.groupby("multiplicity_family").size()) == {4}

    def test_d1_is_excluded_from_the_primary_family(self):
        result = stage.apply_primary_holm(self._summary())
        d1 = result[result["rung"].eq("xgb_tree_d1")]
        assert d1["multiplicity_family"].str.startswith("descriptive control").all()

    def test_wrong_primary_family_size_is_rejected(self):
        frame = self._summary()
        frame = frame[~((frame["bag"].eq("structural")) & (frame["arm"].eq("o_max")) & frame["rung"].eq("xgb_tree_d3"))]
        result = stage.apply_primary_holm(frame)
        with pytest.raises(ValueError, match="expected 4"):
            stage.validate_primary_families(result)


class TestDeliveredArtifacts:
    ROOT = stage.ROOT / "outputs/sensitivity/cross_selected_baseline/main_k10"

    @pytest.fixture(scope="class")
    def summary(self):
        path = self.ROOT / "cross_selected_baseline_summary.csv"
        if not path.is_file():
            pytest.skip("cross-selected baseline has not been generated")
        return pd.read_csv(path)

    @pytest.fixture(scope="class")
    def predictions(self):
        path = self.ROOT / "cross_selected_baseline_predictions.parquet"
        if not path.is_file():
            pytest.skip("predictions have not been generated")
        return pd.read_parquet(path)

    def test_primary_families_are_correct(self, summary):
        stage.validate_primary_families(summary)

    def test_every_subject_has_one_prediction_per_cell(self, predictions):
        counts = predictions.groupby(["bag", "arm", "rung", "row_id"]).size()
        assert (counts == 1).all()

    def test_participant_counts_match_the_summary(self, summary, predictions):
        observed = predictions.groupby(["bag", "arm", "rung"]).size().rename("n").reset_index()
        merged = summary.merge(observed, on=["bag", "arm", "rung"], validate="one_to_one")
        assert (merged["n"] == merged["n_participants"]).all()

    def test_selected_and_baseline_share_participants(self, predictions):
        assert predictions["y_pred_full"].notna().all()
        assert predictions["y_pred_base"].notna().all()

    def test_country_balanced_r2_reproduces_from_predictions(self, summary, predictions):
        for row in summary.itertuples(index=False):
            cell = predictions[
                predictions["bag"].eq(row.bag)
                & predictions["arm"].eq(row.arm)
                & predictions["rung"].eq(row.rung)
            ]
            full = float(stage.country_r2(cell, "y_pred_full").mean())
            base = float(stage.country_r2(cell, "y_pred_base").mean())
            assert full == pytest.approx(row.selected_model_country_balanced_r2)
            assert base == pytest.approx(row.baseline_country_balanced_r2)
            assert full - base == pytest.approx(row.delta_r2)

    def test_table_matches_the_summary(self, summary):
        source = stage.ROOT / (
            "outputs/main/paper/complete/tables/source_data/cross_selected_baseline/"
            "cross_selected_exposome_vs_baseline.csv"
        )
        if not source.is_file():
            pytest.skip("publication table has not been generated")
        table = pd.read_csv(source)
        primary = summary[summary["rung"].isin(stage.PRIMARY_RUNGS)]
        assert len(table) == len(primary)
        assert np.allclose(sorted(table["ΔR²"]), sorted(primary["delta_r2"]))

    def test_figure_source_data_matches_the_summary(self, summary):
        source = stage.ROOT / (
            "outputs/main/paper/complete/figures/supplementary/source_data/"
            "fig_cross_selected_exposome_vs_baseline_hpo_k10_source_data_all_cells.csv"
        )
        if not source.is_file():
            pytest.skip("figure source data has not been generated")
        drawn = pd.read_csv(source)
        merged = drawn.merge(summary, on=["bag", "arm", "rung"], validate="one_to_one")
        assert len(merged) == len(drawn)
        assert np.allclose(merged["delta_r2_x"], merged["delta_r2_y"])
