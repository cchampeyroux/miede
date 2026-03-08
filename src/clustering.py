# -*- coding: utf-8 -*-
"""A helper module for building PDL features and clustering.

All heavy data processing lives inside functions so that an import is fast.
Calling ``get_features_pdl()`` triggers the work lazily and caches the result.
When the module is executed as a script it also runs the clustering analysis and
writes label files to ``../data``.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

# persistent cache for feature computation
from joblib import Memory
_cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
memory = Memory(_cache_dir, verbose=0)

# constants
COL_PDL = "ID"
COL_DT = "horodate"
COL_PWR = "valeur"
DATA_PATHS = ["data/RES2-6-9.csv", "../data/RES2-6-9.csv"]

# cluster -> label mapping
cluster_to_label = {
    0: 0,
    3: 0,
    5: 0,
    8: 0,
    9: 0,
    1: 1,
    2: 1,
    4: 1,
    6: 1,
    7: 1,
}



# ----------------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------------

def _locate_file() -> str:
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("raw data file not found")

@memory.cache
def _read_raw() -> pd.DataFrame:
    path = _locate_file()
    print("in")
    df = pd.read_csv(path, sep=";", parse_dates=[COL_DT])
    print("out")
    df[COL_DT] = pd.to_datetime(df[COL_DT], utc=True)
    df["date"] = df[COL_DT].dt.date
    df["hour"] = df[COL_DT].dt.hour + df[COL_DT].dt.minute / 60
    df["dow"] = df[COL_DT].dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5
    df["hh_index"] = ((df[COL_DT].dt.hour * 60) + df[COL_DT].dt.minute) // 30
    return df


def _compute_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.assign(energy_kwh_step=df[COL_PWR] * 0.5)
        .groupby([COL_PDL, "date"], as_index=False)
        .agg(
            daily_kwh=("energy_kwh_step", "sum"),
            daily_mean_kw=(COL_PWR, "mean"),
            daily_max_kw=(COL_PWR, "max"),
            n_steps=(COL_PWR, "size"),
        )
    )
    def q20_positive(s: pd.Series) -> float:
        s = s[s > 0]
        if len(s) == 0:
            return np.nan
        return s.quantile(0.2)
    daily["th_pdl"] = daily.groupby(COL_PDL)["daily_kwh"].transform(q20_positive)
    daily["is_active_day"] = (daily["daily_kwh"] >= daily["th_pdl"]).fillna(False)
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    daily["dow"] = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["is_weekend"] = daily["dow"] >= 5
    
    def season(m: int) -> str:
        if m in (12,1,2):
            return "winter"
        if m in (6,7,8):
            return "summer"
        return "mid"
    daily["season"] = daily["month"].apply(season)
    return daily


def _compute_activity(daily: pd.DataFrame) -> pd.DataFrame:
    return (
        daily.groupby(COL_PDL, as_index=False)
        .agg(
            active_day_rate=("is_active_day", "mean"),
            mean_daily_kwh=("daily_kwh", "mean"),
            p95_daily_kwh=("daily_kwh", lambda s: np.percentile(s,95)),
            cv_daily_kwh=("daily_kwh", lambda s: s.std()/(s.mean()+1e-9)),
        )
    )


def _runs_and_gaps(s: pd.Series) -> pd.Series:
    runs = []
    gaps = []
    run = gap = 0
    for v in s:
        if v:
            run += 1
            if gap > 0:
                gaps.append(gap)
                gap = 0
        else:
            gap += 1
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    if gap > 0:
        gaps.append(gap)
    return pd.Series({
        "n_runs": len(runs),
        "mean_run_len": float(np.mean(runs)) if runs else 0.0,
        "max_run_len": float(np.max(runs)) if runs else 0.0,
        "mean_gap_len": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_len": float(np.max(gaps)) if gaps else 0.0,
    })


def _compute_runs_stats(daily: pd.DataFrame) -> pd.DataFrame:
    return (
        daily.sort_values([COL_PDL,"date"])
        .groupby(COL_PDL)["is_active_day"]
        .apply(_runs_and_gaps)
        .unstack()
        .reset_index()
    )


def _compute_week_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    week_pattern = (
        daily.groupby([COL_PDL,"is_weekend"], as_index=False)
        .agg(active_rate=("is_active_day","mean"), mean_kwh=("daily_kwh","mean"))
        .pivot(index=COL_PDL, columns="is_weekend")
    )
    week_pattern.columns = [f"{a}_{'weekend' if b else 'weekday'}" for a,b in week_pattern.columns]
    return week_pattern.reset_index()


def _compute_season_stats(daily: pd.DataFrame) -> pd.DataFrame:
    daily2 = daily.copy()
    daily2["date_ts"] = pd.to_datetime(daily2["date"])
    daily2["month"] = daily2["date_ts"].dt.month
    def season_from_month(m:int)->str:
        if m in (12,1,2): return "winter"
        if m in (6,7,8): return "summer"
        return "mid"
    daily2["season"] = daily2["month"].map(season_from_month)
    season_stats = (
        daily2.groupby([COL_PDL,"season"], as_index=False)
        .agg(mean_daily_kwh=("daily_kwh","mean"))
        .pivot(index=COL_PDL,columns="season",values="mean_daily_kwh")
        .reset_index()
    )
    for c in ["winter","summer","mid"]:
        if c not in season_stats.columns:
            season_stats[c]=0.0
    global_mean = (
        daily2.groupby(COL_PDL,as_index=False)
        .agg(mean_daily_kwh_global=("daily_kwh","mean"))
    )
    season_stats = season_stats.merge(global_mean,on=COL_PDL,how="left",validate="one_to_one")
    eps=1e-9
    season_stats["r_global"] = 1.0
    season_stats["r_mid"] = season_stats["mid"]/(season_stats["mean_daily_kwh_global"]+eps)
    season_stats["r_summer"] = season_stats["summer"]/(season_stats["mean_daily_kwh_global"]+eps)
    season_stats["r_winter"] = season_stats["winter"]/(season_stats["mean_daily_kwh_global"]+eps)
    return season_stats[[COL_PDL,"r_global","r_mid","r_summer","r_winter"]]


def _assemble_features(daily: pd.DataFrame) -> pd.DataFrame:
    activity = _compute_activity(daily)
    runs_stats = _compute_runs_stats(daily)
    week_pattern = _compute_week_pattern(daily)
    season_stats = _compute_season_stats(daily)
    features = (
        activity
        .merge(runs_stats, on=COL_PDL, how="left", validate="one_to_one")
        .merge(week_pattern, on=COL_PDL, how="left", validate="one_to_one")
        .merge(season_stats, on=COL_PDL, how="left", validate="one_to_one")
    )
    features["seasonality_amp"] = features[["r_mid","r_summer","r_winter"]].max(axis=1) - features[["r_mid","r_summer","r_winter"]].min(axis=1)
    features["winter_minus_summer"] = features["r_winter"] - features["r_summer"]
    assert features[COL_PDL].is_unique, "one row per PDL expected"
    return features

feature_cols = [
    "active_day_rate","n_runs","mean_run_len","max_run_len",
    "mean_gap_len","max_gap_len",
    "mean_daily_kwh","p95_daily_kwh","cv_daily_kwh",
    "active_rate_weekday","active_rate_weekend",
    "mean_kwh_weekday","mean_kwh_weekend",
    "winter_minus_summer","seasonality_amp",
    "r_global","r_mid","r_summer","r_winter",
]

@memory.cache
def _build_features() -> pd.DataFrame:
    """Compute and return features_pdl from raw data.

    This is the expensive work that we want to persist.  ``joblib`` will
    cache the output to disk; subsequent calls (even in a new Python
    process) will reuse the stored result unless inputs change.
    """
    raw = _read_raw()
    print(raw.head())
    daily = _compute_daily(raw)
    return _assemble_features(daily)


def get_features_pdl(force: bool = False) -> pd.DataFrame:
    """Public accessor for PDL features.

    Parameters
    ----------
    force : bool, default=False
        If True the cache is cleared before recomputing.  This is handy if
        the underlying raw data has been updated and you want to refresh
        the cached result.
    """
    if force:
        # clear the joblib cache for all functions
        memory.clear(warn=False)
    return _build_features().copy()

def compute_clusters(features_pdl: pd.DataFrame, n_clusters:int=10, random_state:int=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = features_pdl[feature_cols].copy()
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters,n_init=50,random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    out = features_pdl.copy()
    out["cluster"] = labels
    return out, kmeans

def create_label_df(features_with_cluster: pd.DataFrame) -> pd.DataFrame:
    labels = features_with_cluster[[COL_PDL,"cluster"]].copy()
    labels["label"] = labels["cluster"].map(cluster_to_label)
    return labels.rename(columns={COL_PDL:"id"})

def save_labels(features_with_cluster: pd.DataFrame, path: str=None) -> str:
    df = create_label_df(features_with_cluster)
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir,"..","data","RES2-6-9_labels_bis.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path,index=False)
    return path

if __name__ == "__main__":
    feats = get_features_pdl()
    feats_clustered, km = compute_clusters(feats, n_clusters=10)
    print("Computed", len(feats_clustered), "features with clusters")
    save_path = save_labels(feats_clustered)
    print(f"Labels saved into {save_path}")
