"""Model features, one row per outlet and hour.

Every feature for a given hour uses only sales from at least 7 days earlier,
so the same features work for every day of a 7-day-ahead forecast without
feeding forecasts back in, and the back-test cannot peek at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS = (7, 14, 28)
HOLIDAY_DISTANCE_CAP = 14  # "more than two weeks away" all looks the same

FEATURES = [
    "outlet", "hour", "weekday", "is_public_holiday", "is_school_holiday", "days_to_ph", "days_from_ph",
    "rain_mm", "max_temp_c", "lag_7d", "lag_14d", "lag_28d", "mean_4w_same_hour", "daily_mean_7d",
    "lag_7d_was_holiday",
]


def build_grid(sales: pd.DataFrame, history_end: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """Every (outlet, hour) the outlet has ever traded in, for every day of history and the forecast window.

    Hours with no sales row in history count as 0 transactions; future hours are NaN.
    """
    start = sales["timestamp"].min().normalize()
    dates = pd.date_range(start, history_end + pd.Timedelta(days=horizon_days), freq="D")
    parts = []
    for outlet, hours in sales.groupby("outlet")["timestamp"].apply(lambda t: sorted(t.dt.hour.unique())).items():
        ts = (dates.to_numpy()[:, None] + (np.array(hours) * 3600).astype("timedelta64[s]")[None, :]).ravel()
        parts.append(pd.DataFrame({"timestamp": ts, "outlet": outlet}))
    grid = pd.concat(parts, ignore_index=True).merge(sales, on=["timestamp", "outlet"], how="left")
    history = grid["timestamp"] < history_end + pd.Timedelta(days=1)
    grid.loc[history, "transactions"] = grid.loc[history, "transactions"].fillna(0)
    return grid.sort_values(["outlet", "timestamp"]).reset_index(drop=True)


def holiday_distances(dates: pd.Series, holiday_dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Days until the next and since the last public holiday, capped."""
    d = dates.to_numpy().astype("datetime64[D]")
    h = np.sort(holiday_dates.to_numpy().astype("datetime64[D]"))
    if len(h) == 0:
        cap = np.full(len(d), HOLIDAY_DISTANCE_CAP)
        return cap, cap
    nxt = np.searchsorted(h, d, side="left")
    prv = np.searchsorted(h, d, side="right") - 1
    to_ph = np.where(nxt < len(h), (h[np.minimum(nxt, len(h) - 1)] - d).astype(int), HOLIDAY_DISTANCE_CAP)
    from_ph = np.where(prv >= 0, (d - h[np.maximum(prv, 0)]).astype(int), HOLIDAY_DISTANCE_CAP)
    return np.minimum(to_ph, HOLIDAY_DISTANCE_CAP), np.minimum(from_ph, HOLIDAY_DISTANCE_CAP)


def add_features(grid: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    df = grid.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday

    cal = signals[["date", "is_public_holiday", "is_school_holiday", "rain_mm", "max_temp_c"]].copy()
    cal["days_to_ph"], cal["days_from_ph"] = holiday_distances(cal["date"], cal.loc[cal["is_public_holiday"] == 1, "date"])
    cal_prev = cal[["date", "is_public_holiday"]].assign(date=cal["date"] + pd.Timedelta(days=7))
    cal = cal.merge(cal_prev.rename(columns={"is_public_holiday": "lag_7d_was_holiday"}), on="date", how="left")
    df = df.merge(cal, on="date", how="left")
    if df["is_public_holiday"].isna().any():
        missing = df.loc[df["is_public_holiday"].isna(), "date"]
        raise ValueError(f"signals_daily.csv does not cover {missing.min():%Y-%m-%d} to {missing.max():%Y-%m-%d}; rerun `make signals`")

    # Same hour, k days earlier.
    y = df.set_index(["outlet", "timestamp"])["transactions"]
    for k in (7, 14, 21, 28):
        idx = pd.MultiIndex.from_arrays([df["outlet"], df["timestamp"] - pd.Timedelta(days=k)])
        df[f"lag_{k}d"] = y.reindex(idx).to_numpy()
    df["mean_4w_same_hour"] = df[["lag_7d", "lag_14d", "lag_21d", "lag_28d"]].mean(axis=1)

    # Average daily total over the 7 days that ended a week ago: the outlet's recent level.
    daily = df.groupby(["outlet", "date"])["transactions"].sum(min_count=1)
    level = daily.groupby(level="outlet").transform(lambda s: s.rolling(7, min_periods=7).mean().shift(7))
    df = df.merge(level.rename("daily_mean_7d").reset_index(), on=["outlet", "date"], how="left")

    df["outlet"] = pd.Categorical(df["outlet"], categories=sorted(df["outlet"].unique()))
    return df
