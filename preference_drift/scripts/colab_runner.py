"""
Colab-friendly runner for the preference drift experiment.

Usage in Google Colab:
  Cell 1: Install dependencies
    !pip install transformers datasets torch pyyaml matplotlib seaborn scipy tqdm

  Cell 2: Clone or upload the project
    !git clone <your-repo-url> preference_drift
    %cd preference_drift

  Cell 3: Run the experiment
    from scripts.colab_runner import run_in_colab
    output_dir, results = run_in_colab()

  Cell 4: Display figures inline
    from scripts.colab_runner import show_figures
    show_figures(output_dir)
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_in_colab(config_path: str = None):
    """Run the full experiment pipeline. Returns (output_dir, drift_results)."""
    if config_path is None:
        config_path = str(project_root / "configs" / "experiments" / "gpt2_math_crowspairs.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.pipeline import run_pipeline
    return run_pipeline(config_path)


def show_figures(output_dir):
    """Display all generated figures inline in a Colab notebook."""
    from IPython.display import Image, display

    figures_dir = Path(output_dir) / "figures"
    for fig_path in sorted(figures_dir.glob("*.png")):
        print(f"\n--- {fig_path.name} ---")
        display(Image(filename=str(fig_path)))
