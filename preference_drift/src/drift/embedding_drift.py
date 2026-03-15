"""Embedding space drift analysis: track how concept representations shift after fine-tuning."""

import torch
import torch.nn.functional as F
from typing import Dict, Any


def extract_concept_embeddings(
    model,
    concept_pairs: Dict[str, list],
    template: str = "This is about {word}.",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract embeddings for all concept words organized by category.

    Returns: {category: {word: embedding_tensor}}
    """
    embeddings = {}
    for category, pairs in concept_pairs.items():
        embeddings[category] = {}
        for pair in pairs:
            for word in pair:
                embeddings[category][word] = model.get_word_embedding(word, template)
    return embeddings


def compute_embedding_drift(
    pre_embeddings: Dict[str, Dict[str, torch.Tensor]],
    post_embeddings: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Compare embeddings before and after fine-tuning.

    Computes:
    - Per-word cosine drift (1 - cosine_similarity)
    - Per-pair similarity shift (how pair relationships changed)
    """
    per_word = {}
    pair_drift = {}

    for category in pre_embeddings:
        per_word[category] = {}
        pair_drift[category] = {}

        # Per-word drift
        for word in pre_embeddings[category]:
            pre_emb = pre_embeddings[category][word]
            post_emb = post_embeddings[category][word]
            cos_sim = F.cosine_similarity(pre_emb.unsqueeze(0), post_emb.unsqueeze(0)).item()
            per_word[category][word] = float(1.0 - cos_sim)

        # Per-pair relationship drift
        # Reconstruct pairs from the flat word dict
        words = list(pre_embeddings[category].keys())
        for i in range(0, len(words) - 1, 2):
            w1, w2 = words[i], words[i + 1]
            pair_name = f"{w1}-{w2}"

            pre_sim = F.cosine_similarity(
                pre_embeddings[category][w1].unsqueeze(0),
                pre_embeddings[category][w2].unsqueeze(0),
            ).item()
            post_sim = F.cosine_similarity(
                post_embeddings[category][w1].unsqueeze(0),
                post_embeddings[category][w2].unsqueeze(0),
            ).item()

            pair_drift[category][pair_name] = {
                "pre_similarity": float(pre_sim),
                "post_similarity": float(post_sim),
                "delta": float(post_sim - pre_sim),
            }

    # Summary statistics
    all_word_drifts = [
        d for cat in per_word.values() for d in cat.values()
    ]

    return {
        "per_word": per_word,
        "pair_drift": pair_drift,
        "summary": {
            "mean_word_drift": float(sum(all_word_drifts) / len(all_word_drifts)) if all_word_drifts else 0.0,
            "max_word_drift": float(max(all_word_drifts)) if all_word_drifts else 0.0,
            "min_word_drift": float(min(all_word_drifts)) if all_word_drifts else 0.0,
        },
    }
