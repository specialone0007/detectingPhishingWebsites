"""Download, cache and load the UCI Phishing Websites dataset (id 327).

11,055 websites, 30 hand-crafted features in {-1, 0, 1}, one label.
Feature semantics follow Mohammad, Thabtah & McCluskey (2012/2014): for every feature
-1 is the "phishing-looking" value and 1 the "legitimate-looking" value; 0 is "suspicious".
The label column `Result` uses the same convention (-1 = phishing, 1 = legitimate). We expose
it as `is_phishing` in {0, 1} so that "positive" means "phishing" everywhere in this package.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

UCI_ZIP = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
ARFF_MEMBER = "Training Dataset.arff"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "phishing_websites.csv"

FEATURES: tuple[str, ...] = (
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token",
    "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
    "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe",
    "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank", "Google_Index",
    "Links_pointing_to_page", "Statistical_report",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "Address bar": FEATURES[0:12],
    "Abnormal": FEATURES[12:18],
    "HTML / JavaScript": FEATURES[18:23],
    "Domain": FEATURES[23:30],
}

TARGET = "is_phishing"


def _download() -> pd.DataFrame:
    raw = urlopen(UCI_ZIP, timeout=120).read()  # noqa: S310 - fixed https URL
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(ARFF_MEMBER).decode("utf-8")
    data, _meta = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(data)
    # scipy returns nominal attributes as bytes
    for c in df.columns:
        df[c] = df[c].astype(str).astype(int)
    return df


def load_dataset(cache: Path | str | None = DEFAULT_CACHE, download: bool = True) -> pd.DataFrame:
    """Return the dataset with the 30 features and an `is_phishing` column in {0, 1}.

    Reads `cache` if it exists, otherwise downloads from UCI and writes the cache.
    """
    cache = Path(cache) if cache is not None else None
    if cache is not None and cache.exists():
        df = pd.read_csv(cache)
    else:
        if not download:
            raise FileNotFoundError(f"no cached dataset at {cache} and download=False")
        df = _download()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
    return to_frame(df)


def to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and map `Result` (-1 phishing / 1 legitimate) to `is_phishing`."""
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns: {missing}")
    df = raw[list(FEATURES)].astype(np.int8).copy()
    if TARGET in raw.columns:
        df[TARGET] = raw[TARGET].astype(np.int8)
    elif "Result" in raw.columns:
        df[TARGET] = (raw["Result"].astype(int) == -1).astype(np.int8)
    else:
        raise ValueError("dataset needs a `Result` or `is_phishing` column")
    bad = ~df[list(FEATURES)].isin([-1, 0, 1]).all(axis=None)
    if bad:
        raise ValueError("feature values outside {-1, 0, 1}")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[list(FEATURES)], df[TARGET]


def train_test(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out split. The test fold is touched exactly once, at the very end."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
