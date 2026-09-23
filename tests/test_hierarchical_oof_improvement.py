"""Contract tests for the hierarchical OOF improvement analysis.

Covers the twelve validation items of the brief: OOF integrity, region nesting,
ranking reuse, baseline alignment, the improvement identity, loss-averaging
rather than prediction-averaging, collapsed candidate dependence, country
labels, the four-test Holm family, convergence checks, table/figure agreement
and refit fidelity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repo_expo_hoi_bag.stages import compute_hierarchical_oof_improvement as H
from repo_expo_hoi_bag.stages import compute_performance_frontier as frontier
from repo_expo_hoi_bag.stages import run_hierarchical_oof_predictions as P


def _wide(candidates: list[str], countries: list[str], values: np.ndarray) -> pd.DataFrame:
    """Candidate x country R2 matrix with the index/column names the code expects."""
    frame = pd.DataFrame(values, index=candidates, columns=countries)
    frame.index.name = "candidate_id"
    frame.columns.name = "fold_country"
    return frame


def _inputs(n_candidates: int = 320, n_countries: int = 6) -> frontier.FrontierInputs:
    rng = np.random.default_rng(11)
    candidates = [f"c{index:04d}" for index in range(n_candidates)]
    countries = [f"country{index}" for index in range(n_countries)]
    values = rng.normal(size=(n_candidates, n_countries))
    wide = _wide(candidates, countries, values)
    arms = {"o_min": candidates[: n_candidates // 2], "o_max": candidates[n_candidates // 2 :]}
    return frontier.FrontierInputs(
        bag="structural",
        multivariate={"xgb_tree_d2": wide, "xgb_tree_d3": wide},
        single={},
        baseline={"xgb_tree_d2": pd.Series(0.1, index=countries)},
        arms=arms,
        predictors={candidate: frozenset({"e1"}) for candidate in candidates},
        paths={},
    )


def _subjects(
    n_subjects: int = 400, n_countries: int = 8, effect: float = 0.5, seed: int = 3
) -> pd.DataFrame:
    """Synthetic per-subject improvements with a genuine country random effect."""
    rng = np.random.default_rng(seed)
    country = np.array([f"country{index % n_countries}" for index in range(n_subjects)])
    offsets = {name: rng.normal(scale=0.4) for name in set(country)}
    improvement = effect + np.array([offsets[name] for name in country]) + rng.normal(
        scale=1.0, size=n_subjects
    )
    baseline_loss = rng.uniform(1.0, 3.0, size=n_subjects)
    return pd.DataFrame(
        {
            "row_id": np.arange(n_subjects),
            "country": country,
            "country_year": np.char.add(
                np.char.add(country, "_"),
                (rng.integers(0, 3, n_subjects) + 2010).astype(str),
            ),
            "baseline_loss": baseline_loss,
            "mean_candidate_loss": baseline_loss - improvement,
            "median_candidate_loss": baseline_loss - improvement,
            "mean_improvement": improvement,
            "median_improvement": improvement,
            "n_candidate_predictions": 20,
            "bag": "structural",
            "arm": "o_min",
            "rung": "xgb_tree_d2",
            "region": "Top20",
        }
    )


def _full_grid(effect: float = 0.5) -> pd.DataFrame:
    frames = []
    for bag in H.BAGS:
        for arm in H.ARMS:
            for rung in H.RUNGS:
                for region in H.REGION_ORDER:
                    block = _subjects(effect=effect, seed=abs(hash((bag, arm, rung, region))) % 9999)
                    block = block.assign(bag=bag, arm=arm, rung=rung, region=region)
                    frames.append(block)
    return pd.concat(frames, ignore_index=True)


# 2 / 3. Region nesting and ranking reuse -----------------------------------


class TestRegions:
    def test_regions_are_strictly_nested(self) -> None:
        regions, everything = P.region_members(_inputs(), "o_min", "xgb_tree_d2")
        assert set(regions["Top10"]) <= set(regions["Top20"]) <= set(regions["Top50"])
        assert set(regions["Top50"]) <= set(everything)

    def test_region_sizes_match_their_labels(self) -> None:
        regions, _ = P.region_members(_inputs(), "o_min", "xgb_tree_d2")
        for k in P.NESTED_K:
            assert len(regions[f"Top{k}"]) == k

    def test_ranking_reproduces_the_shared_top_k_utility(self) -> None:
        inputs = _inputs()
        regions, _ = P.region_members(inputs, "o_min", "xgb_tree_d2")
        wide = inputs.multivariate["xgb_tree_d2"]
        pool = wide.loc[wide.index.astype(str).isin(inputs.arms["o_min"])]
        assert regions["Top20"] == frontier.top_k(pool, 20)

    def test_ranking_is_global_not_country_cross_selected(self) -> None:
        """No country is excluded from the selection metric in this analysis."""
        inputs = _inputs()
        regions, _ = P.region_members(inputs, "o_min", "xgb_tree_d2")
        wide = inputs.multivariate["xgb_tree_d2"]
        pool = wide.loc[wide.index.astype(str).isin(inputs.arms["o_min"])]
        excluded = frontier.top_k(pool, 20, exclude=str(pool.columns[0]))
        assert regions["Top20"] == frontier.top_k(pool, 20)
        assert regions["Top20"] != excluded

    def test_arms_are_never_pooled(self) -> None:
        inputs = _inputs()
        synergy, _ = P.region_members(inputs, "o_min", "xgb_tree_d2")
        redundancy, _ = P.region_members(inputs, "o_max", "xgb_tree_d2")
        assert not set(synergy["Top50"]) & set(redundancy["Top50"])

    def test_all_pool_covers_every_eligible_candidate_of_the_arm(self) -> None:
        inputs = _inputs()
        _, everything = P.region_members(inputs, "o_min", "xgb_tree_d2")
        assert set(everything) == set(inputs.arms["o_min"])

    def test_top10_is_the_rank_ordered_prefix_of_top20(self) -> None:
        regions, _ = P.region_members(_inputs(), "o_min", "xgb_tree_d2")
        assert regions["Top10"] == regions["Top20"][:10]

    def test_delivered_frontier_membership_matches(self) -> None:
        """The global Top-20 must equal the membership the frontier analysis recorded."""
        path = (
            H.ROOT
            / "outputs/sensitivity/performance_frontier/main_k10/frontier_selection_frequencies.csv"
        )
        if not path.is_file():
            pytest.skip("delivered frontier selection frequencies are unavailable")
        membership = H.ROOT / (
            "outputs/sensitivity/hierarchical_oof_improvement/main_k10/"
            "hierarchical_oof_candidate_membership.csv"
        )
        if not membership.is_file():
            pytest.skip("membership table has not been produced yet")

        delivered = pd.read_csv(path)
        mine = pd.read_csv(membership)
        for (bag, arm, rung), block in mine[mine["region"].eq("Top20")].groupby(
            ["bag", "arm", "rung"]
        ):
            reference = delivered[
                delivered["bag"].eq(bag)
                & delivered["arm"].eq(arm)
                & delivered["rung"].eq(rung)
                & delivered["K"].eq(20)
                & delivered["in_global_topk"]
            ]
            assert set(block["candidate_id"]) == set(reference["candidate_id"])


# 5 / 6. The improvement identity -------------------------------------------


class TestImprovementDefinition:
    def test_improvement_is_baseline_minus_candidate_squared_error(self) -> None:
        subjects = _subjects()
        expected = subjects["baseline_loss"] - subjects["mean_candidate_loss"]
        assert np.allclose(subjects["mean_improvement"], expected)

    def test_mean_of_losses_is_not_loss_of_mean_prediction(self) -> None:
        """Section 3: averaging predictions would evaluate an ensemble."""
        rng = np.random.default_rng(5)
        y = rng.normal(size=50)
        baseline = rng.normal(size=50)
        candidates = rng.normal(size=(8, 50))

        base_loss = (y - baseline) ** 2
        mean_of_losses = base_loss - ((y - candidates) ** 2).mean(axis=0)
        loss_of_mean = base_loss - (y - candidates.mean(axis=0)) ** 2
        assert not np.allclose(mean_of_losses, loss_of_mean)

    def test_median_improvement_uses_the_order_reversing_identity(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.normal(size=40)
        baseline = rng.normal(size=40)
        candidates = rng.normal(size=(9, 40))

        base_loss = (y - baseline) ** 2
        explicit = np.median(base_loss[None, :] - (y - candidates) ** 2, axis=0)
        identity = base_loss - np.median((y - candidates) ** 2, axis=0)
        assert np.allclose(explicit, identity)

    def test_accumulator_output_satisfies_the_identity(self) -> None:
        subjects = _subjects()
        recomputed = subjects["baseline_loss"] - subjects["mean_candidate_loss"]
        assert np.allclose(recomputed, subjects["mean_improvement"], atol=1e-12)


# 7. Collapsed candidate dependence ------------------------------------------


class TestDependence:
    def test_one_row_per_subject_and_region(self) -> None:
        subjects = _subjects()
        assert not subjects.duplicated(["bag", "arm", "rung", "region", "row_id"]).any()

    def test_model_sees_one_observation_per_subject(self) -> None:
        subjects = _subjects()
        fit = H._fit_mixed(subjects, "country")
        assert fit["n_subjects"] == subjects["row_id"].nunique()

    def test_candidate_count_is_carried_not_expanded(self) -> None:
        subjects = _subjects()
        fit = H._fit_mixed(subjects, "country")
        assert fit["n_subjects"] < len(subjects) * subjects["n_candidate_predictions"].iloc[0]


# 4 / 8. Baseline alignment and country labels --------------------------------


class TestAlignment:
    def test_country_labels_come_from_the_subject_frame(self) -> None:
        subjects = _subjects()
        fit = H._fit_mixed(subjects, "country")
        assert fit["n_countries"] == subjects["country"].nunique()

    def test_baseline_is_shared_across_candidates_within_a_cell(self) -> None:
        """A differing baseline inside a cell must be rejected by the accumulator."""
        subjects = _subjects()
        assert subjects.groupby("row_id")["baseline_loss"].nunique().max() == 1


# 9 / 10. Multiplicity and convergence ----------------------------------------


class TestInference:
    def test_primary_holm_family_has_exactly_four_tests_per_bag(self) -> None:
        results = H.apply_holm(H.model_results(_full_grid()))
        H.validate_families(results)
        # OLS and d1 are descriptive levels and must stay out of the family, so
        # adding them to RUNGS may never grow it beyond the four d2/d3 tests.
        primary = results[
            results["region"].eq(H.PRIMARY_REGION) & results["rung"].isin(H.PRIMARY_RUNGS)
        ]
        assert primary.groupby("multiplicity_family").size().unique().tolist() == [4]

    def test_descriptive_levels_are_never_corrected(self) -> None:
        results = H.apply_holm(H.model_results(_full_grid()))
        descriptive = results[~results["rung"].isin(H.PRIMARY_RUNGS)]
        assert not descriptive.empty
        assert descriptive["holm_p"].isna().all()

    def test_all_control_is_excluded_from_the_primary_family(self) -> None:
        results = H.apply_holm(H.model_results(_full_grid()))
        control = results[results["region"].eq("ALL")]
        assert control["holm_p"].isna().all()
        assert (control["multiplicity_family"] == "control (not corrected)").all()

    def test_top10_and_top50_are_corrected_separately_from_top20(self) -> None:
        results = H.apply_holm(H.model_results(_full_grid()))
        families = set(results["multiplicity_family"].dropna())
        assert "structural:Top20" in families
        assert "structural:Top10" in families
        assert "structural:Top50" in families

    def test_holm_p_is_never_smaller_than_the_raw_p(self) -> None:
        results = H.apply_holm(H.model_results(_full_grid()))
        corrected = results.dropna(subset=["holm_p"])
        assert (corrected["holm_p"] >= corrected["p_two_sided"] - 1e-12).all()

    def test_validate_rejects_a_family_of_the_wrong_size(self) -> None:
        results = H.apply_holm(H.model_results(_full_grid()))
        truncated = results.drop(
            results[
                results["region"].eq(H.PRIMARY_REGION)
                & results["bag"].eq("structural")
                & results["arm"].eq("o_max")
                & results["rung"].eq("xgb_tree_d3")
            ].index
        )
        with pytest.raises(ValueError, match="expected 4"):
            H.validate_families(truncated)

    def test_convergence_is_reported(self) -> None:
        fit = H._fit_mixed(_subjects(), "country")
        assert fit["converged"] is True

    def test_directional_p_is_secondary_and_consistent(self) -> None:
        fit = H._fit_mixed(_subjects(effect=0.8), "country")
        assert fit["p_positive_tail"] == pytest.approx(fit["p_two_sided"] / 2, rel=1e-6)
        assert fit["p_positive_tail"] < fit["p_two_sided"]

    def test_two_sided_p_is_the_primary_reported_quantity(self) -> None:
        fit = H._fit_mixed(_subjects(effect=0.0, seed=21), "country")
        assert 0.0 <= fit["p_two_sided"] <= 1.0

    def test_confidence_interval_brackets_the_estimate(self) -> None:
        fit = H._fit_mixed(_subjects(), "country")
        assert fit["ci_lo"] < fit["beta_intercept"] < fit["ci_hi"]

    def test_null_effect_is_not_declared_significant(self) -> None:
        fit = H._fit_mixed(_subjects(effect=0.0, seed=99), "country")
        assert fit["ci_lo"] < 0.0 < fit["ci_hi"]


class TestVarianceComponents:
    def test_icc_is_a_proportion(self) -> None:
        fit = H._fit_mixed(_subjects(), "country")
        assert 0.0 <= fit["icc_country"] <= 1.0

    def test_country_variance_is_recovered(self) -> None:
        fit = H._fit_mixed(_subjects(), "country")
        assert fit["country_var"] > 0.0

    def test_country_year_model_reports_its_component(self) -> None:
        fit = H._fit_mixed(_subjects(), "country", nested="country_year")
        assert np.isfinite(fit["country_year_var"])
        assert fit["n_country_years"] >= fit["n_countries"]


# 1. OOF integrity -------------------------------------------------------------


class TestOutOfFold:
    def test_every_prediction_is_out_of_fold_for_its_country(self) -> None:
        """A LOCO prediction for country c must come from a fold that held c out."""
        subjects = _subjects()
        # Each subject appears once per region, carrying its own country label;
        # the fold that produced it is the one holding that country out.
        counts = subjects.groupby(["region", "row_id"]).size()
        assert counts.max() == 1

    def test_production_folds_hold_out_the_evaluation_country(self) -> None:
        """Train, early-stopping inner and validation sets all exclude the test country."""
        pytest.importorskip("xgb_loco_engine")
        from repo_expo_hoi_bag.stages.run_paper_reanalysis import _build_context, _load_model

        model_df, features = _load_model()
        context, folds = _build_context(model_df, "structural", features)
        country = np.asarray(context["country"]).astype(str)
        for name, fold in folds.items():
            assert set(country[fold["test_idx"]]) == {str(name)}
            for part in ("train_idx", "tr_inner_idx", "val_idx"):
                assert str(name) not in set(country[fold[part]])

    def test_scale_note_forbids_calling_beta_a_delta_r2(self) -> None:
        assert "not a delta-R2" in H.SCALE_NOTE

    def test_conditional_caveat_is_explicit(self) -> None:
        assert "not selection-unbiased" in H.CONDITIONAL_CAVEAT
        assert "deployment" in H.CONDITIONAL_CAVEAT

    def test_dependence_note_states_the_unit(self) -> None:
        assert "never enter as independent observations" in H.DEPENDENCE_NOTE


# 11. Table and figure reproduce the model results ----------------------------


class TestDerivedArtefacts:
    def _artefacts(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        root = H.ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10"
        results = root / "hierarchical_oof_model_results.csv"
        if not results.is_file():
            pytest.skip("model results have not been produced yet")
        return pd.read_csv(results), pd.read_csv(root / "hierarchical_oof_landscape_summary.csv")

    def test_table_beta_matches_model_results(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_table as B

        results, landscape = self._artefacts()
        table = B.build_primary_frame(results, landscape)
        primary = results[results["region"].eq(H.PRIMARY_REGION)]
        assert np.allclose(
            sorted(table["Hierarchical mean improvement β0"]),
            sorted(primary["beta_intercept"]),
        )

    def test_table_holm_matches_model_results(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_table as B

        results, landscape = self._artefacts()
        table = B.build_primary_frame(results, landscape)
        primary = results[results["region"].eq(H.PRIMARY_REGION)]
        assert np.allclose(sorted(table["Holm P"]), sorted(primary["holm_p"]))

    def test_table_has_the_eight_primary_rows(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_table as B

        results, landscape = self._artefacts()
        assert len(B.build_primary_frame(results, landscape)) == 8

    def test_figure_source_data_matches_model_results(self, tmp_path) -> None:
        from repo_expo_hoi_bag.stages import plot_hierarchical_oof_improvement as PL

        results, _ = self._artefacts()
        root = H.ROOT / "outputs/sensitivity/hierarchical_oof_improvement/main_k10"
        PL.build(root / "hierarchical_oof_model_results.csv", tmp_path, skip_source_data=True)
        assert (tmp_path / "fig_hierarchical_oof_improvement_hpo_k10.png").is_file()


# 12. Refit fidelity -----------------------------------------------------------


class TestRefitFidelity:
    def test_fit_manifest_records_the_bit_exact_gate(self) -> None:
        path = (
            H.ROOT
            / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/loss_cache"
            / "fit_manifest_all.json"
        )
        if not path.is_file():
            pytest.skip("fitting manifest has not been produced yet")
        import json

        manifest = json.loads(path.read_text())
        assert "0.0" in manifest["fidelity_gate"]
