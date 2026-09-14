"""Figures for the README. Matplotlib only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402

from phishing.data import FEATURES, GROUPS  # noqa: E402
from phishing.models import PRETTY  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
BLUE, RED, GREY, INK = "#1f4e79", "#c0392b", "#95a5a6", "#2c3e50"
GROUP_COLOURS = dict(zip(GROUPS, ["#1f4e79", "#c0392b", "#27ae60", "#8e44ad"], strict=True))


def _group_of(feature: str) -> str:
    for g, fs in GROUPS.items():
        if feature in fs:
            return g
    return "?"


def plot_class_balance(counts: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.bar(counts.index, counts.values, color=[BLUE, RED], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{v / counts.sum():.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("websites")
    ax.set_title("Class balance", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_agreement(scores: pd.Series, out: Path) -> None:
    s = scores.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    colours = [GROUP_COLOURS[_group_of(f)] for f in s.index]
    ax.barh(s.index, s.values, color=colours)
    ax.axvline(0.5, color=GREY, ls="--", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("single-feature rule agrees with label (fraction of non-abstaining rows)")
    ax.set_title("How good is each feature on its own?", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOURS.values()]
    ax.legend(handles, GROUP_COLOURS.keys(), frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_importance(importances: pd.Series, out: Path, title: str) -> None:
    s = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.barh(s.index, s.values, color=[GROUP_COLOURS[_group_of(f)] for f in s.index])
    ax.set_xlabel("total split gain (normalised)")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out: Path) -> None:
    corr = df[list(FEATURES)].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Pearson r")
    ax.set_title("Feature correlations", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
    for name, (y_true, score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, lw=1.3, label=PRETTY.get(name, name))
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC on the held-out test fold (zoomed)", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, out: Path) -> None:
    r = results.sort_values("test_accuracy")
    labels = [PRETTY.get(m, m) for m in r["model"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(r))
    ax.barh(y - 0.18, r["cv_accuracy_mean"], height=0.36, color=GREY, label="5-fold CV (train)")
    ax.errorbar(r["cv_accuracy_mean"], y - 0.18, xerr=r["cv_accuracy_std"], fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=2)
    ax.barh(y + 0.18, r["test_accuracy"], height=0.36, color=BLUE, label="held-out test")
    for yi, v in zip(y, r["test_accuracy"], strict=True):
        ax.text(v + 0.002, yi + 0.18, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("accuracy")
    ax.set_title("Model comparison", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["legitimate", "phishing"], cmap="Blues", ax=ax,
        colorbar=False,
    )
    ax.grid(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
