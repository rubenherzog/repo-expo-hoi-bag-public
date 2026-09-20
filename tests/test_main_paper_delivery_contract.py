from pathlib import Path

import pandas as pd
import pytest
import yaml

from repo_expo_hoi_bag.stages.verify_main_paper_delivery import (
    _validate_whole_pca_source_data,
)


ROOT = Path(__file__).resolve().parents[1]


def test_main_paper_delivery_contract_includes_set_size_sensitivity() -> None:
    config = yaml.safe_load((ROOT / "config" / "main_paper_delivery.yaml").read_text())
    assert config["delivery_root"] == "outputs/main/paper/complete"
    assert "fig5_generative_account" in config["excluded"]
    assert "fig_s10_set_size_sensitivity" in config["supplementary_figures"]
    assert "fig_s10_set_size_sensitivity" not in config["excluded"]
    assert "combined_bag_outputs" in config["excluded"]
    assert len(config["supplementary_tables"]["sheets"]) == 18


def test_main_paper_delivery_verifier_requires_local_main_root() -> None:
    text = (ROOT / "src/repo_expo_hoi_bag/stages/verify_main_paper_delivery.py").read_text()
    assert 'config["delivery_root"]' in text
    assert "REPRO_DATA_ROOT" not in text


def test_main_paper_collector_only_accepts_lightweight_delivery_files() -> None:
    text = (ROOT / "src/repo_expo_hoi_bag/stages/collect_main_paper_delivery.py").read_text()
    assert 'ALLOWED_SUFFIXES = {".csv", ".json", ".pdf", ".png", ".svg", ".tiff", ".txt", ".xlsx"}' in text
    assert "REPRO_DATA_ROOT" not in text


def test_whole_pca_delivery_rejects_bag_specific_variance(tmp_path: Path) -> None:
    source = tmp_path / "figures/supplementary/source_data"
    source.mkdir(parents=True)
    structural = pd.DataFrame(
        {
            "pc_n": [1, 2],
            "explained_variance_ratio": [0.46, 0.20],
            "cumulative_explained_variance_ratio": [0.46, 0.66],
        }
    )
    functional = structural.copy()
    structural.to_csv(
        source / "whole_exposome_pca_sensitivity_source_data_a1_struct_pca_variance.csv",
        index=False,
    )
    functional.to_csv(
        source / "whole_exposome_pca_sensitivity_source_data_b1_func_pca_variance.csv",
        index=False,
    )
    _validate_whole_pca_source_data(tmp_path)

    functional.loc[0, "explained_variance_ratio"] = 0.50
    functional.to_csv(
        source / "whole_exposome_pca_sensitivity_source_data_b1_func_pca_variance.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="variance differs by BAG"):
        _validate_whole_pca_source_data(tmp_path)
