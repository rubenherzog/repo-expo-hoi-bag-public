"""Contract tests for the hierarchical bootstrap of the Top-K incremental R2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from repo_expo_hoi_bag.stages import compute_hierarchical_oof_bootstrap as B


def _cell(n_countries: int = 6, n_years: int = 3, n_subjects: int = 40, effect: float = 0.4,
          seed: int = 5) -> pd.DataFrame:
    """Synthetic cell with a genuine country-year layer and a positive effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for country_index in range(n_countries):
        country = f"country{country_index}"
        for year in range(n_years):
            y = rng.normal(scale=3.0, size=n_subjects)
            base = (y - rng.normal(scale=2.0, size=n_subjects)) ** 2
            rows.append(
                pd.DataFrame(
                    {
                        "row_id": rng.integers(0, 10**9, n_subjects),
                        "country": country,
                        "country_year": f"{country}_{2010 + year}",
                        "y_true": y,
                        "baseline_loss": base,
                        "mean_candidate_loss": base - effect,
                        "n_candidate_predictions": 20,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


class TestObservedEstimand:
    def test_delta_r2_matches_its_definition(self) -> None:
        frame = _cell()
        observed = B.observed_delta_r2(frame)
        block = frame[frame["country"].eq("country0")]
        y = block["y_true"].to_numpy(float)
        sst = ((y - y.mean()) ** 2).sum()
        expected = (block["baseline_loss"].sum() - block["mean_candidate_loss"].sum()) / sst
        row = observed[observed["country"].eq("country0")].iloc[0]
        assert row["delta_r2_country"] == pytest.approx(expected)

    def test_one_row_per_country(self) -> None:
        frame = _cell()
        assert len(B.observed_delta_r2(frame)) == frame["country"].nunique()

    def test_constant_improvement_gives_positive_delta(self) -> None:
        observed = B.observed_delta_r2(_cell(effect=0.5))
        assert (observed["delta_r2_country"] > 0).all()

    def test_zero_improvement_gives_zero_delta(self) -> None:
        observed = B.observed_delta_r2(_cell(effect=0.0))
        assert np.allclose(observed["delta_r2_country"], 0.0)


class TestResampling:
    def test_countries_are_held_fixed(self) -> None:
        """Every replicate must span exactly the observed countries."""
        frame = _cell()
        blocks, owners = B._country_year_blocks(frame)
        assert set(owners) == set(frame["country"].unique())

    def test_country_year_blocks_partition_the_rows(self) -> None:
        frame = _cell()
        blocks, _ = B._country_year_blocks(frame)
        covered = np.sort(np.concatenate(blocks))
        assert np.array_equal(covered, np.arange(len(frame)))

    def test_each_block_is_one_country_year(self) -> None:
        frame = _cell()
        blocks, owners = B._country_year_blocks(frame)
        for block, owner in zip(blocks, owners):
            rows = frame.iloc[block]
            assert rows["country_year"].nunique() == 1
            assert set(rows["country"]) == {owner}

    def test_bootstrap_is_deterministic_under_a_fixed_seed(self) -> None:
        frame = _cell()
        first, _ = B.bootstrap_cell(frame, draws=64, seed=7)
        second, _ = B.bootstrap_cell(frame, draws=64, seed=7)
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_draws(self) -> None:
        frame = _cell()
        first, _ = B.bootstrap_cell(frame, draws=64, seed=7)
        second, _ = B.bootstrap_cell(frame, draws=64, seed=8)
        assert not np.array_equal(first, second)

    def test_results_do_not_depend_on_worker_count(self) -> None:
        """Each replicate carries its own seed, so n_jobs must not change results."""
        frame = _cell()
        sequential, _ = B.bootstrap_cell(frame, draws=96, seed=99, n_jobs=1)
        for workers in (4, 8):
            parallel, _ = B.bootstrap_cell(frame, draws=96, seed=99, n_jobs=workers)
            assert np.array_equal(sequential, parallel)

    def test_replicate_count_is_honoured(self) -> None:
        estimates, _ = B.bootstrap_cell(_cell(), draws=128, seed=3)
        assert len(estimates) == 128

    def test_interval_brackets_a_real_effect(self) -> None:
        estimates, observed = B.bootstrap_cell(_cell(effect=0.5), draws=400, seed=11)
        summary = B.summarise(estimates, observed)
        assert summary["ci_lo"] > 0
        assert summary["p_one_sided"] < 0.05

    def test_null_effect_is_not_significant(self) -> None:
        estimates, observed = B.bootstrap_cell(_cell(effect=0.0), draws=400, seed=13)
        summary = B.summarise(estimates, observed)
        assert summary["ci_lo"] <= 0 <= summary["ci_hi"]


class TestSummary:
    def test_p_value_is_never_exactly_zero(self) -> None:
        estimates = np.full(1000, 5.0)
        summary = B.summarise(estimates, B.observed_delta_r2(_cell()))
        assert summary["p_one_sided"] > 0

    def test_fraction_positive_is_a_proportion(self) -> None:
        estimates, observed = B.bootstrap_cell(_cell(), draws=128, seed=3)
        summary = B.summarise(estimates, observed)
        assert 0.0 <= summary["fraction_bootstrap_positive"] <= 1.0

    def test_ci_brackets_the_bootstrap_median(self) -> None:
        estimates, observed = B.bootstrap_cell(_cell(), draws=256, seed=3)
        summary = B.summarise(estimates, observed)
        assert summary["ci_lo"] <= summary["bootstrap_median"] <= summary["ci_hi"]


class TestMultiplicity:
    def _results(self) -> pd.DataFrame:
        rows = []
        for bag in ("structural", "functional"):
            for arm in ("o_min", "o_max"):
                for rung in ("xgb_tree_d2", "xgb_tree_d3"):
                    for region in ("Top5", "Top20", "Top50", "ALL"):
                        rows.append(
                            {"bag": bag, "arm": arm, "rung": rung, "region": region,
                             "p_one_sided": 0.01}
                        )
        return pd.DataFrame(rows)

    def test_primary_family_is_four_tests_per_bag(self) -> None:
        results = B.apply_holm(self._results())
        primary = results[results["region"].eq("Top20")]
        assert primary.groupby("multiplicity_family").size().unique().tolist() == [4]

    def test_grid_is_not_corrected(self) -> None:
        results = B.apply_holm(self._results())
        grid = results[results["region"].ne("Top20")]
        assert grid["holm_p"].isna().all()
        assert (grid["multiplicity_family"] == "sensitivity grid (not corrected)").all()

    def test_holm_is_never_smaller_than_raw(self) -> None:
        results = B.apply_holm(self._results())
        corrected = results.dropna(subset=["holm_p"])
        assert (corrected["holm_p"] >= corrected["p_one_sided"] - 1e-12).all()

    def test_wrong_family_size_is_rejected(self) -> None:
        results = self._results()
        trimmed = results.drop(
            results[
                results["region"].eq("Top20")
                & results["bag"].eq("structural")
                & results["arm"].eq("o_max")
                & results["rung"].eq("xgb_tree_d3")
            ].index
        )
        with pytest.raises(ValueError, match="expected 4"):
            B.apply_holm(trimmed)


class TestNoRefitting:
    def test_manifest_states_zero_refitting(self) -> None:
        assert "no model is refitted" in B.RESAMPLING_NOTE

    def test_k_grid_is_not_a_selection_device(self) -> None:
        assert "not used to select K" in B.K_GRID_NOTE

    def test_region_sort_puts_all_last(self) -> None:
        regions = ["ALL", "Top100", "Top5", "Top20"]
        assert sorted(regions, key=B._region_sort_key) == ["Top5", "Top20", "Top100", "ALL"]


class TestConsolidatedOutputs:
    """The table and figure must reproduce the bootstrap results exactly."""

    def _results(self) -> pd.DataFrame:
        path = (
            B.ROOT
            / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap"
            / "hierarchical_oof_bootstrap_results.csv"
        )
        if not path.is_file():
            pytest.skip("bootstrap results have not been produced yet")
        return pd.read_csv(path)

    def test_primary_table_has_the_eight_rows(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_bootstrap_table as T

        assert len(T.build_primary_frame(self._results())) == 8

    def test_primary_table_matches_the_results(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_bootstrap_table as T

        results = self._results()
        table = T.build_primary_frame(results)
        primary = results[results["region"].eq("Top20")]
        assert np.allclose(
            sorted(table["ΔR² over baseline"]), sorted(primary["observed_delta_r2"])
        )
        assert np.allclose(sorted(table["Holm P"]), sorted(primary["holm_p"]))

    def test_grid_table_covers_every_region(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_bootstrap_table as T

        results = self._results()
        grid = T.build_grid_frame(results)
        assert len(grid) == len(results)
        assert set(grid["Candidate region"]) == set(results["region"])

    def test_only_top20_rows_carry_a_holm_value(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_bootstrap_table as T

        grid = T.build_grid_frame(self._results())
        assert grid.loc[grid["Candidate region"].ne("Top20"), "Holm P"].isna().all()
        assert grid.loc[grid["Candidate region"].eq("Top20"), "Holm P"].notna().all()

    def test_grid_marks_primary_and_control_roles(self) -> None:
        from repo_expo_hoi_bag.stages import build_hierarchical_oof_bootstrap_table as T

        grid = T.build_grid_frame(self._results())
        assert set(grid.loc[grid["Candidate region"].eq("Top20"), "Role"]) == {"Primary"}
        assert set(grid.loc[grid["Candidate region"].eq("ALL"), "Role"]) == {"Control"}

    def test_figure_uses_the_repository_style_contract(self) -> None:
        from repo_expo_hoi_bag.stages import plot_hierarchical_oof_bootstrap as P
        from repo_expo_hoi_bag.figures import style

        assert P.ARM_COLOR["o_min"] == style.SYN_COLOR
        assert P.ARM_COLOR["o_max"] == style.RED_COLOR
        # Level spellings come from the single source of truth, not local strings.
        assert style.LEVEL_LABELS["xgb_tree_d3"] == "d3"

    def test_figure_renders_every_format(self, tmp_path) -> None:
        from repo_expo_hoi_bag.stages import plot_hierarchical_oof_bootstrap as P

        root = (
            B.ROOT
            / "outputs/sensitivity/hierarchical_oof_improvement/main_k10/bootstrap"
            / "hierarchical_oof_bootstrap_results.csv"
        )
        if not root.is_file():
            pytest.skip("bootstrap results have not been produced yet")
        P.build(root, tmp_path, skip_source_data=True)
        for extension in ("png", "pdf", "svg", "tiff"):
            assert (tmp_path / f"fig_hierarchical_oof_bootstrap_k_curve.{extension}").is_file()
