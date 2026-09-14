"""Cross-validated model selection on the training fold, one final fit on the test fold."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from phishing.models import PRETTY, make_model


@dataclass
class Result:
    model: str
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: float
    fit_seconds: float

    @property
    def pretty(self) -> str:
        return PRETTY.get(self.model, self.model)


def _scores(est: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def evaluate_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    folds: int = 5,
    seed: int = 42,
) -> tuple[Result, BaseEstimator]:
    est = make_model(name, seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cvres = cross_validate(clone(est), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    pred = est.predict(X_test)
    res = Result(
        model=name,
        cv_accuracy_mean=float(cvres["test_score"].mean()),
        cv_accuracy_std=float(cvres["test_score"].std()),
        test_accuracy=float(accuracy_score(y_test, pred)),
        test_precision=float(precision_score(y_test, pred)),
        test_recall=float(recall_score(y_test, pred)),
        test_f1=float(f1_score(y_test, pred)),
        test_roc_auc=float(roc_auc_score(y_test, _scores(est, X_test))),
        fit_seconds=fit_s,
    )
    return res, est


def results_frame(results: list[Result]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in results])
    return df.sort_values("test_accuracy", ascending=False, ignore_index=True)
