"""Custom greedy O-information search (project-specific).

This is the only genuinely custom piece of the original vendored ``thoi_branch``
fork: a greedy search that deduplicates repeat solutions at each order step, plus
an O/S-ratio objective variant. It depends on the upstream ``thoi`` package (pip)
for all O-information primitives.

Supported THOI releases expose one of two collector contracts: unpacked measures
(``collector(nplets, tc, dtc, o, s, bn)``) or a stacked ``batch_result`` tensor of
shape ``(B, D, 4)`` ordered ``[tc, dtc, o, s]``. The compatibility adapter below
normalizes both contracts before calling ``thoi.collectors.batch_to_tensor`` so
candidate selection is unchanged.

Moved verbatim from thoi_branch/thoi/heuristics/greedy_no_repeats.py.
"""
from typing import Union, Callable, List, Optional, Dict, Any
from functools import partial
import time

import numpy as np
import torch
from tqdm import tqdm

from thoi.typing import TensorLikeArray
from thoi.measures.gaussian_copula import multi_order_measures
from thoi.collectors import batch_to_tensor, concat_batched_tensors
from thoi.commons import _normalize_input_data
from thoi.heuristics.greedy import _next_order_greedy


def _os_ratio_metric(batched_res: Union[torch.Tensor, np.ndarray], eps: float = 1e-12):
    """Return the mean O/S ratio across datasets with zero-safe fallback.

    The ratio is computed from the aggregated O and S measures so it remains
    consistent with THOI's current per-metric averaging semantics.
    """
    if isinstance(batched_res, torch.Tensor):
        o = batched_res[:, :, 2].mean(dim=1).to(dtype=torch.float32)
        s = batched_res[:, :, 3].mean(dim=1).to(dtype=torch.float32)
        safe = torch.abs(s) > float(eps)
        return torch.where(safe, o / s, torch.zeros_like(o))

    arr = np.asarray(batched_res)
    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise ValueError(f"Expected batched measures with shape (batch, D, 4), got {arr.shape}.")
    o = arr[:, :, 2].mean(axis=1).astype(np.float32, copy=False)
    s = arr[:, :, 3].mean(axis=1).astype(np.float32, copy=False)
    ratio = np.zeros_like(o, dtype=np.float32)
    safe = np.abs(s) > float(eps)
    ratio[safe] = o[safe] / s[safe]
    return torch.as_tensor(ratio, dtype=torch.float32)


def _batch_to_tensor_compatible(
    nplets,
    *collector_args,
    top_k,
    metric,
    largest,
):
    """Forward either supported THOI collector contract to ``batch_to_tensor``.

    THOI 0.2.40 calls custom collectors with the public six-argument contract
    ``(nplets, tc, dtc, o, s, batch_number)``. THOI 0.2.42 calls them with the
    internal stacked contract ``(nplets, batch_result, batch_number)``, where the
    final axis of ``batch_result`` is ``[tc, dtc, o, s]``. Supporting both here
    keeps the greedy selection identical while allowing the project environment
    and the pinned publication environment to use the same implementation.
    """
    if len(collector_args) == 2:
        batch_result, batch_number = collector_args
        if batch_result.ndim != 3 or batch_result.shape[-1] != 4:
            raise ValueError(
                "Expected stacked THOI measures with shape (batch, datasets, 4), "
                f"got {tuple(batch_result.shape)}."
            )
        tc, dtc, o, s = (batch_result[..., index] for index in range(4))
    elif len(collector_args) == 5:
        tc, dtc, o, s, batch_number = collector_args
    else:
        raise TypeError(
            "Unsupported THOI collector contract: expected 2 or 5 arguments "
            f"after nplets, received {len(collector_args)}."
        )

    return batch_to_tensor(
        nplets,
        tc,
        dtc,
        o,
        s,
        batch_number,
        top_k=top_k,
        metric=metric,
        largest=largest,
    )


def greedy_no_repeats_os_ratio(*args, **kwargs):
    """Convenience wrapper for greedy no-repeats using the O/S ratio objective."""
    kwargs = dict(kwargs)
    kwargs["metric"] = _os_ratio_metric
    return greedy_no_repeats(*args, **kwargs)

@torch.no_grad()
def greedy_no_repeats(
    X: TensorLikeArray,
    initial_order: int = 3,
    order: Optional[int] = None,
    *,
    covmat_precomputed: bool = False,
    T: Optional[Union[int, List[int]]] = None,
    repeat: int = 10,
    batch_size: int = 1000000,
    repeat_batch_size: int = 1000000,
    device: torch.device = torch.device("cpu"),
    metric: Union[str, Callable] = "o",
    largest: bool = False,
    return_profile: bool = False,
):
    """
    Greedy equivalent to thoi.heuristics.greedy.greedy, but removes duplicated
    repeat solutions at each order step before evaluating next candidates.
    """

    covmats, D, N, T = _normalize_input_data(X, covmat_precomputed, T, device)

    # Initial solutions (same as original greedy). The collector adapter accepts
    # both supported THOI contracts before forwarding to batch_to_tensor.
    batch_data_collector = partial(
        _batch_to_tensor_compatible,
        top_k=repeat,
        metric=metric,
        largest=largest,
    )
    batch_aggregation = partial(concat_batched_tensors, top_k=repeat, metric=None, largest=largest)

    _, current_solution, current_scores = multi_order_measures(
        covmats,
        covmat_precomputed=True,
        T=T,
        min_order=initial_order,
        max_order=initial_order,
        batch_size=batch_size,
        device=device,
        batch_data_collector=batch_data_collector,
        batch_aggregation=batch_aggregation,
    )

    # Keep path order for correct prefix export at each order.
    current_solution = current_solution.to(device).contiguous()
    order = order if order is not None else N

    best_scores = [current_scores]
    profile: List[Dict[str, Any]] = []

    order_iterator = list(range(initial_order + 1, order + 1))
    order_pbar = tqdm(order_iterator, desc="Order", leave=False)
    iter_times: List[float] = []

    for next_order in order_pbar:
        t0 = time.perf_counter()

        # Canonical set representation for deduplication only.
        # Path representation in current_solution is preserved for output prefixes.
        canonical_solution, _ = torch.sort(current_solution, dim=1)
        unique_solution, inverse_idx = torch.unique(
            canonical_solution, dim=0, return_inverse=True
        )
        n_total = int(current_solution.shape[0])
        n_unique = int(unique_solution.shape[0])

        # Evaluate only unique states
        best_candidate_u, best_score_u = _next_order_greedy(
            covmats,
            T,
            unique_solution,
            metric=metric,
            largest=largest,
            batch_size=batch_size,
            repeat_batch_size=repeat_batch_size,
            device=device,
        )

        # Map back to all repeats
        best_candidate = best_candidate_u[inverse_idx]
        best_score = best_score_u[inverse_idx]

        best_scores.append(best_score)

        # Update path solution (prefix order preserved).
        current_solution = torch.cat((current_solution, best_candidate.unsqueeze(1)), dim=1)

        elapsed = time.perf_counter() - t0
        iter_times.append(elapsed)
        avg_iter = sum(iter_times) / len(iter_times)
        rem = len(order_iterator) - len(iter_times)
        eta = avg_iter * rem
        order_pbar.set_postfix(
            {
                "last_s": f"{elapsed:.2f}",
                "avg_s": f"{avg_iter:.2f}",
                "eta_s": f"{eta:.1f}",
                "uniq": f"{n_unique}/{n_total}",
            }
        )

        if return_profile:
            profile.append(
                {
                    "order": int(next_order),
                    "n_repeat_total": n_total,
                    "n_repeat_unique": n_unique,
                    "dedup_fraction": float(1.0 - (n_unique / n_total if n_total else 0.0)),
                    "elapsed_sec": float(elapsed),
                }
            )

    out_scores = torch.stack(best_scores).T
    if return_profile:
        return current_solution, out_scores, profile
    return current_solution, out_scores
