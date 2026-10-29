"""Rebuild the corrected oracle figure from released aggregate results."""
from pathlib import Path
from minibatch_apo.paper.generate_paper_figures import build_oracle_pruning_figure

def main():
    root = Path(__file__).resolve().parents[2]
    for path in build_oracle_pruning_figure(root, root / "figures", root / "results/corrected_ablations"):
        print(f"Wrote {path}")

if __name__ == "__main__":
    main()
