"""Visualization functions for preference drift experiment results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def plot_stereotype_pct_comparison(
    drift_result: Dict[str, Any], output_path: Path, figsize=(12, 6)
):
    """Grouped bar chart: stereotype % pre vs post, per bias category."""
    categories = list(drift_result["by_category"].keys())
    pre_vals = [drift_result["by_category"][c]["stereotype_pct_pre"] for c in categories]
    post_vals = [drift_result["by_category"][c]["stereotype_pct_post"] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, pre_vals, width, label="Pre-FT", color="#4878CF")
    ax.bar(x + width / 2, post_vals, width, label="Post-FT", color="#D65F5F")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Unbiased (50%)")
    ax.set_ylabel("Stereotype Preference (%)")
    ax.set_title("Stereotype Preference Before vs After Fine-Tuning (CrowS-Pairs)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path / "stereotype_pct_comparison.png", dpi=150)
    plt.close(fig)


def plot_drift_heatmap(drift_result: Dict[str, Any], output_path: Path):
    """Heatmap of stereotype_pct_delta per category. Red=more stereo, blue=less."""
    categories = list(drift_result["by_category"].keys())
    deltas = [drift_result["by_category"][c]["stereotype_pct_delta"] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 2))
    data = pd.DataFrame([deltas], columns=categories, index=["Drift"])
    sns.heatmap(
        data, annot=True, fmt=".3f", center=0, cmap="RdBu_r",
        ax=ax, cbar_kws={"label": "Delta stereotype %"}
    )
    ax.set_title("Preference Drift by Category (positive = more stereotypical)")
    fig.tight_layout()
    fig.savefig(output_path / "drift_heatmap.png", dpi=150)
    plt.close(fig)


def plot_per_item_drift_distribution(
    per_item_df: pd.DataFrame, output_path: Path
):
    """Histogram of per-item logprob_diff_delta values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(per_item_df["logprob_diff_delta"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", label="No drift")
    ax.set_xlabel("Log-prob difference delta (post - pre)")
    ax.set_ylabel("Number of items")
    ax.set_title("Distribution of Per-Item Preference Drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "per_item_drift_distribution.png", dpi=150)
    plt.close(fig)


def plot_embedding_drift_bars(embedding_drift: Dict[str, Any], output_path: Path):
    """Bar chart of per-word cosine drift, colored by category."""
    data = []
    for category, words in embedding_drift["per_word"].items():
        for word, drift in words.items():
            data.append({"category": category, "word": word, "drift": drift})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.barplot(data=df, x="word", y="drift", hue="category", ax=ax)
    ax.set_ylabel("Cosine drift (1 - cosine_sim(pre, post))")
    ax.set_title("Per-Concept Embedding Drift After Fine-Tuning")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path / "embedding_drift_bars.png", dpi=150)
    plt.close(fig)


def plot_pair_similarity_shift(embedding_drift: Dict[str, Any], output_path: Path):
    """Dumbbell chart: how cosine similarity between concept pairs changed after FT."""
    data = []
    for category, pairs in embedding_drift["pair_drift"].items():
        for pair_name, vals in pairs.items():
            data.append({
                "pair": f"{category}: {pair_name}",
                "pre": vals["pre_similarity"],
                "post": vals["post_similarity"],
            })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    y_pos = range(len(df))

    for i, row in df.iterrows():
        ax.plot([row["pre"], row["post"]], [i, i], "ko-", markersize=6)
        ax.plot(row["pre"], i, "bo", markersize=8, label="Pre-FT" if i == 0 else "")
        ax.plot(row["post"], i, "ro", markersize=8, label="Post-FT" if i == 0 else "")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["pair"])
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Concept Pair Similarity: Pre vs Post Fine-Tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "pair_similarity_shift.png", dpi=150)
    plt.close(fig)
