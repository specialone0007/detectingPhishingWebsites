"""Unit tests. The dataset tests use a small synthetic frame so CI never needs the network;
`test_real_dataset` runs only when the cached CSV is present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phishing import features
from phishing.data import DEFAULT_CACHE, FEATURES, TARGET, load_dataset, to_frame, train_test
from phishing.evaluate import evaluate_model, results_frame
from phishing.models import MODELS, make_model


def synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Features in {-1,0,1}; the label is mostly decided by the first three features so
    every model has something to learn."""
    rng = np.random.default_rng(seed)
    X = rng.choice([-1, 0, 1], size=(n, len(FEATURES)))
    signal = X[:, 0] + X[:, 1] + X[:, 2] + rng.normal(0, 0.8, n)
    result = np.where(signal < 0, -1, 1)  # -1 = phishing like the UCI file
    df = pd.DataFrame(X, columns=FEATURES)
    df["Result"] = result
    return df


def test_to_frame_maps_result_to_is_phishing():
    df = to_frame(synthetic())
    assert TARGET in df.columns
    assert set(df[TARGET].unique()) <= {0, 1}
    raw = synthetic()
    assert (df[TARGET].to_numpy() == (raw["Result"].to_numpy() == -1)).all()


def test_to_frame_rejects_bad_values():
    raw = synthetic()
    raw.loc[0, FEATURES[0]] = 5
    with pytest.raises(ValueError):
        to_frame(raw)


def test_to_frame_rejects_missing_column():
    raw = synthetic().drop(columns=[FEATURES[3]])
    with pytest.raises(ValueError):
        to_frame(raw)


def test_split_is_stratified():
    df = to_frame(synthetic(1000))
    X_tr, X_te, y_tr, y_te = train_test(df, test_size=0.2, seed=1)
    assert len(X_te) == 200 and len(X_tr) == 800
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_agreement_scores_range_and_signal():
    df = to_frame(synthetic(2000))
    s = features.agreement_scores(df)
    assert s.between(0, 1).all()
    # the three informative features must beat the uninformative ones
    assert s[[FEATURES[0], FEATURES[1], FEATURES[2]]].min() > s[list(FEATURES[5:])].max()


def test_high_correlations_empty_on_independent_features():
    df = to_frame(synthetic(3000))
    assert features.high_correlations(df, threshold=0.5).empty


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_fits_and_predicts(name):
    df = to_frame(synthetic(300))
    X_tr, X_te, y_tr, y_te = train_test(df, seed=0)
    est = make_model(name, seed=0)
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_make_model_unknown_name():
    with pytest.raises(KeyError):
        make_model("not-a-model")


def test_evaluate_model_logistic():
    df = to_frame(synthetic(600))
    X_tr, X_te, y_tr, y_te = train_test(df, seed=0)
    res, _ = evaluate_model("logistic", X_tr, y_tr, X_te, y_te, folds=3, seed=0)
    assert 0.6 < res.test_accuracy <= 1
    assert 0.5 < res.test_roc_auc <= 1
    table = results_frame([res])
    assert list(table["model"]) == ["logistic"]


@pytest.mark.skipif(not DEFAULT_CACHE.exists(), reason="dataset not cached locally")
def test_real_dataset():
    df = load_dataset(download=False)
    assert df.shape == (11055, 31)
    bal = features.class_balance(df)
    assert bal["phishing"] == 4898 and bal["legitimate"] == 6157
