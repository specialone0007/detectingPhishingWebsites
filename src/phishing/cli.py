"""`phishing-experiments`: run the whole study and write results/ and docs/figures/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phishing import features, plots
from phishing.data import FEATURES, load_dataset, train_test
from phishing.evaluate import evaluate_model, results_frame
from phishing.models import MODELS, PRETTY

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    a = p.parse_args(argv)
    a.results.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print(f"dataset: {len(df):,} rows, {len(FEATURES)} features")

    # ---- model-free analysis ------------------------------------------------------
    balance = features.class_balance(df)
    print("class balance:", balance.to_dict())
    plots.plot_class_balance(balance, a.figures / "class-balance.png")

    agree = features.agreement_scores(df)
    agree.to_csv(a.results / "feature-agreement.csv", header=True)
    plots.plot_agreement(agree, a.figures / "feature-agreement.png")

    features.high_correlations(df).to_csv(a.results / "high-correlations.csv", index=False)
    plots.plot_correlation(df, a.figures / "feature-correlation.png")

    # ---- models ---------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test(df, seed=a.seed)
    print(f"train {len(X_train):,} / test {len(X_test):,} (stratified, seed {a.seed})")

    results, curves, fitted = [], {}, {}
    for name in a.models:
        res, est = evaluate_model(name, X_train, y_train, X_test, y_test, a.folds, a.seed)
        results.append(res)
        fitted[name] = est
        proba = est.predict_proba(X_test)[:, 1]
        curves[name] = (y_test.to_numpy(), proba)
        print(f"{PRETTY[name]:<22} cv {res.cv_accuracy_mean:.4f}+-{res.cv_accuracy_std:.4f}  "
              f"test acc {res.test_accuracy:.4f}  f1 {res.test_f1:.4f}  "
              f"auc {res.test_roc_auc:.4f}  fit {res.fit_seconds:5.1f}s", flush=True)

    table = results_frame(results)
    table.to_csv(a.results / "model-results.csv", index=False)
    plots.plot_model_comparison(table, a.figures / "model-comparison.png")
    plots.plot_roc(curves, a.figures / "roc.png")

    best = table.iloc[0]["model"]
    plots.plot_confusion(y_test, fitted[best].predict(X_test), a.figures / "confusion-best.png",
                         f"{PRETTY[best]}: confusion matrix (test)")

    if "lightgbm" in fitted:
        booster = fitted["lightgbm"].booster_
        gain = pd.Series(booster.feature_importance("gain"), index=FEATURES, name="gain")
        gain = gain / gain.sum()
        gain.sort_values(ascending=False).to_csv(a.results / "lightgbm-gain.csv", header=True)
        plots.plot_importance(gain, a.figures / "feature-importance-lightgbm.png",
                              "LightGBM feature importance (split gain)")

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.drop(columns=["fit_seconds"]).to_string(index=False))
    np.save(a.results / "test-index.npy", X_test.index.to_numpy())
    return 0


if __name__ == "__main__":
    sys.exit(main())
