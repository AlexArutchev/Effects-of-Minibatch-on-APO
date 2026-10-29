"""Check oracle figure values and required seed coverage."""

import json

import numpy as np
import pytest

from minibatch_apo.paper import generate_paper_figures


def test_oracle_figure_uses_aggregate_data(tmp_path, monkeypatch):
    for condition, mean in [("real100", -0.004), ("real100_m10", -0.002)]:
        for seed in range(5):
            folder = tmp_path / "outputs/rule_ablations" / f"{condition}_s{seed}"
            folder.mkdir(parents=True)
            (folder / "oracle_prune.json").write_text(json.dumps({"oracle_gain": mean, "sum_neg_estimate": 0.248}))
    captured = []

    def capture(fig, output_dir, stem):
        captured.extend(p.get_height() for p in fig.axes[0].patches)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return []

    monkeypatch.setattr(generate_paper_figures, "_save_figure", capture)
    generate_paper_figures.build_oracle_pruning_figure(tmp_path, tmp_path / "figures")
    np.testing.assert_allclose(captured, [0.248, 0.248, -0.004, -0.002])


def test_oracle_figure_rejects_missing_seed(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_paper_figures.build_oracle_pruning_figure(tmp_path, tmp_path / "figures")
