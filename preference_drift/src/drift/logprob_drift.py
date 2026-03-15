"""Log-probability drift computation: compare per-item scores before and after fine-tuning."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def compute_logprob_drift(
    pre_scores: pd.DataFrame,
    post_scores: pd.DataFrame,
    benchmark_name: str,
) -> Dict[str, Any]:
    """
    Compare per-item scores before and after fine-tuning.
    Returns drift statistics overall, per-category, and per-item.
    """
    merged = pre_scores.merge(
        post_scores, on="id", suffixes=("_pre", "_post")
    )

    if benchmark_name == "crows_pairs":
        return _compute_crows_pairs_drift(merged)

    return {"error": f"Unsupported benchmark: {benchmark_name}"}


def _compute_crows_pairs_drift(merged: pd.DataFrame) -> Dict[str, Any]:
    """Compute drift for CrowS-Pairs benchmark."""
    merged["logprob_diff_delta"] = (
        merged["logprob_diff_post"] - merged["logprob_diff_pre"]
    )

    merged["preference_flipped"] = (
        merged["prefers_stereotype_pre"] != merged["prefers_stereotype_post"]
    )
    merged["became_more_stereotypical"] = (
        ~merged["prefers_stereotype_pre"] & merged["prefers_stereotype_post"]
    )
    merged["became_less_stereotypical"] = (
        merged["prefers_stereotype_pre"] & ~merged["prefers_stereotype_post"]
    )

    stereo_pct_pre = merged["prefers_stereotype_pre"].mean()
    stereo_pct_post = merged["prefers_stereotype_post"].mean()

    # Paired statistical tests
    t_stat, t_pvalue = stats.ttest_rel(
        merged["logprob_diff_pre"], merged["logprob_diff_post"]
    )

    try:
        w_stat, w_pvalue = stats.wilcoxon(
            merged["logprob_diff_pre"], merged["logprob_diff_post"]
        )
    except ValueError:
        # All differences are zero
        w_stat, w_pvalue = 0.0, 1.0

    result = {
        "overall": {
            "stereotype_pct_pre": float(stereo_pct_pre),
            "stereotype_pct_post": float(stereo_pct_post),
            "stereotype_pct_delta": float(stereo_pct_post - stereo_pct_pre),
            "mean_logprob_diff_delta": float(merged["logprob_diff_delta"].mean()),
            "std_logprob_diff_delta": float(merged["logprob_diff_delta"].std()),
            "n_preferences_flipped": int(merged["preference_flipped"].sum()),
            "n_became_more_stereo": int(merged["became_more_stereotypical"].sum()),
            "n_became_less_stereo": int(merged["became_less_stereotypical"].sum()),
            "paired_ttest_statistic": float(t_stat),
            "paired_ttest_pvalue": float(t_pvalue),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_pvalue": float(w_pvalue),
        },
        "by_category": {},
        "per_item": merged,
    }

    for cat, group in merged.groupby("bias_type_pre"):
        cat_stereo_pre = group["prefers_stereotype_pre"].mean()
        cat_stereo_post = group["prefers_stereotype_post"].mean()

        try:
            cat_t, cat_tp = stats.ttest_rel(
                group["logprob_diff_pre"], group["logprob_diff_post"]
            )
        except ValueError:
            cat_t, cat_tp = 0.0, 1.0

        result["by_category"][cat] = {
            "stereotype_pct_pre": float(cat_stereo_pre),
            "stereotype_pct_post": float(cat_stereo_post),
            "stereotype_pct_delta": float(cat_stereo_post - cat_stereo_pre),
            "mean_logprob_diff_delta": float(group["logprob_diff_delta"].mean()),
            "n_preferences_flipped": int(group["preference_flipped"].sum()),
            "paired_ttest_pvalue": float(cat_tp),
            "n_items": len(group),
        }

    return result
