from __future__ import annotations

import numpy as np
import pandas as pd

from repo_expo_hoi_bag.stages.compute_hpo_model_comparisons import _bootstrap, _sensitivity_tests


def test_country_bootstrap_and_wilcoxon_keep_the_paired_direction() -> None:
    frame = pd.DataFrame(
        {
            "country": np.repeat(["a", "b", "c", "d", "e", "f"], 2),
            "y_true": np.arange(1.0, 13.0),
            "prediction_a": np.arange(1.0, 13.0),
            "prediction_b": np.arange(1.5, 13.5),
        }
    )

    bootstrap = _bootstrap(frame, draws=200, seed=1, column_a="prediction_a", column_b="prediction_b")
    sensitivity = _sensitivity_tests(frame)

    assert bootstrap["delta_r2"] > 0
    assert bootstrap["ci_lo"] > 0
    assert sensitivity["wilcoxon_subject_p_sq"] < 0.05
    assert sensitivity["wilcoxon_country_p_sq"] < 0.05
