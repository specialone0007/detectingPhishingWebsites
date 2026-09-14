"""Phishing website detection on the UCI Phishing Websites dataset."""

from phishing.data import FEATURES, GROUPS, load_dataset, train_test
from phishing.models import MODELS, make_model

__all__ = ["FEATURES", "GROUPS", "MODELS", "load_dataset", "make_model", "train_test"]
