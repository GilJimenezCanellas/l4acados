"""The 'balanced' admission gate of OnlineLearningStrategy.

Run directly (no pytest needed):
    python test/gpytorch_model/test_online_learning_gate.py
"""
import math

import numpy as np
import torch

from l4acados.models.pytorch_models.gpytorch_models.gpytorch_data_processing_strategy import (
    OnlineLearningStrategy,
)


def admitted(points, min_dist=0.15, scales=None, max_num_points=10_000,
             check_invariant=False):
    """Replay the gate over `points`, returning the resulting dictionary rows.

    With `check_invariant`, also assert at every admission that the new point really
    was at least `min_dist` away under the metric in force at that moment. That is
    the contract; it is not the same as "the final dictionary has no pair closer
    than min_dist", because with a running scale estimate the metric keeps moving
    while the first samples come in.
    """
    strat = OnlineLearningStrategy(
        max_num_points=max_num_points, min_dist=min_dist, feature_scales=scales
    )
    kept = []
    for p in points:
        x = torch.as_tensor(p, dtype=torch.float64).reshape(1, -1)
        strat._observe(x)
        if not kept:
            kept.append(x)
            continue
        D = torch.cat(kept, dim=0)
        accept, drop = strat._balanced_gate_and_drop(D, x)
        if not accept:
            continue
        if check_invariant:
            scale = strat._feature_scale(D)
            d = torch.cdist(x / scale, D / scale, p=2).min() / math.sqrt(D.shape[-1])
            assert float(d) > min_dist, f"admitted a point only {float(d)} away"
        if len(kept) >= max_num_points and drop is not None:
            kept.pop(int(drop))
        kept.append(x)
    return torch.cat(kept, dim=0)


def test_threshold_is_absolute_not_seeded_by_the_first_pair():
    """The bug: dictionary size used to depend on the first two samples' spacing.

    Same trajectory sampled at three rates. With an absolute threshold the count is
    set by how much ground the trajectory covers, so all three land close together;
    the old self-referential rule collapsed as the sampling got coarser.
    """
    t = np.linspace(0.0, 20.0, 4000)
    traj = np.stack([np.sin(t), np.cos(3 * t), 0.3 * t], axis=1)

    sizes = [admitted(traj[::stride]).shape[0] for stride in (1, 2, 5)]
    assert min(sizes) > 0.6 * max(sizes), f"still rate dependent: {sizes}"

    # ...and independent of which sample happens to arrive first
    offsets = [admitted(traj[off::5]).shape[0] for off in (0, 1, 2, 37)]
    assert min(offsets) > 0.7 * max(offsets), f"still seed dependent: {offsets}"


def test_admitted_points_respect_min_dist():
    """Every admission clears min_dist under the metric in force at the time."""
    rng = np.random.default_rng(0)
    admitted(rng.normal(size=(2000, 4)), min_dist=0.4, check_invariant=True)


def test_min_dist_controls_dictionary_size():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(2000, 4))
    sizes = [admitted(pts, min_dist=m).shape[0] for m in (0.1, 0.3, 0.6)]
    assert sizes[0] > sizes[1] > sizes[2], sizes


def test_fixed_scales_give_an_exact_final_bound():
    """With scales supplied up front the metric never moves, so the final
    dictionary really does have no pair closer than min_dist."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(2000, 4))
    min_dist = 0.4
    D = admitted(pts, min_dist=min_dist, scales=np.ones(4))
    d = torch.cdist(D, D, p=2) / math.sqrt(D.shape[-1])
    d.fill_diagonal_(float("inf"))
    assert float(d.min()) > min_dist - 1e-9, f"closest pair {float(d.min())}"


def test_scale_free_in_feature_units():
    """Rescaling one feature must not change what gets admitted."""
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(1500, 3))
    blown = pts * np.array([1.0, 1000.0, 1.0])
    assert admitted(pts).shape[0] == admitted(blown).shape[0]


def test_duplicates_are_rejected():
    pts = np.repeat(np.array([[1.0, 2.0, 3.0]]), 50, axis=0)
    assert admitted(pts).shape[0] == 1


def test_explicit_scales_override_the_running_estimate():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(800, 2))
    tight = admitted(pts, scales=np.array([0.1, 0.1]))   # small scale -> big distances
    loose = admitted(pts, scales=np.array([10.0, 10.0]))
    assert tight.shape[0] > loose.shape[0], (tight.shape, loose.shape)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all gate tests passed")
