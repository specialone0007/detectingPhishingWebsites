"""Model-free feature analysis: class balance, agreement scores, correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phishing.data import FEATURES, TARGET


def class_balance(df: pd.DataFrame) -> pd.Series:
    """Counts of legitimate (0) and phishing (1) sites."""
    return df[TARGET].value_counts().sort_index().rename({0: "legitimate", 1: "phishing"})


def agreement_scores(df: pd.DataFrame) -> pd.Series:
    """Per-feature agreement with the label, in [0, 1].

    Each feature already encodes a rule of thumb: -1 says "looks phishing", 1 says "looks
    legitimate", 0 abstains. The score is the fraction of *non-abstaining* rows where that
    rule agrees with the true label, i.e. how good the single-feature rule is as a
    classifier on its own. 0.5 is a coin flip; values well below 0.5 mean the rule is
    informative but inverted. This is the "feature validity value" idea from
    Zhu et al. (OFS-NN, IEEE Access 2019) written as one number per feature.
    """
    y_sign = np.where(df[TARGET].to_numpy() == 1, -1, 1)  # phishing -> -1 like the features
    out = {}
    for f in FEATURES:
        x = df[f].to_numpy()
        mask = x != 0
        out[f] = float((x[mask] == y_sign[mask]).mean()) if mask.any() else float("nan")
    return pd.Series(out, name="agreement").sort_values(ascending=False)


def high_correlations(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Feature pairs whose absolute Pearson correlation exceeds `threshold`."""
    corr = df[list(FEATURES)].corr()
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) > threshold:
                rows.append({"feature_a": a, "feature_b": b, "pearson_r": round(float(r), 3)})
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "pearson_r"])
    return out.sort_values("pearson_r", key=abs, ascending=False, ignore_index=True)


def label_correlation(df: pd.DataFrame) -> pd.Series:
    """Point-biserial correlation of each feature with `is_phishing`, most negative first
    (a strongly negative value means: low feature value goes with phishing, as designed)."""
    return df[list(FEATURES)].corrwith(df[TARGET]).sort_values()
