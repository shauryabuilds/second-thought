"""Hourly demand forecast per outlet, and an honest back-test against today's rules.

Baselines (what operators do today)
  last_week_rule   same hour, same day last week
  four_week_avg    average of the same hour on the same weekday over the last 4 weeks

Model
  One LightGBM model across all outlets (outlet is a category feature), trained
  on calendar, holiday, weather and lagged-sales features (kaizen/features.py).
  Two extra quantile models give an 80% range (lower, upper).

Back-test
  Rolling origin over the last 8 weeks: stand at the end of week N, train on
  everything before, forecast the next 7 days, move on a week. Accuracy is
  1 - WAPE (total absolute error / total actual transactions), per outlet.

Writes to the client's outputs folder:
  forecast.csv          timestamp, outlet, forecast, lower, upper  (the next 7 days, hourly)
  backtest.csv          hourly back-test predictions for the model and both baselines
  backtest_accuracy.csv 1 - WAPE per outlet, method and level (hour, day)
  daily_sales.csv       date, outlet, actual, forecast, last_week_rule, rain_mm, public_holiday
"""

from __future__ import annotations

import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from kaizen.config import HORIZON_DAYS, ClientConfig, load_client, setup_logging
from kaizen.contract import load_inputs
from kaizen.features import FEATURES, add_features, build_grid
from kaizen.signals import load_signals

log = logging.getLogger(__name__)

BACKTEST_WEEKS = 8
MIN_TRAIN_DAYS = 56
METHODS = {"model": "forecast", "last_week_rule": "last_week_rule", "four_week_avg": "four_week_avg"}
PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_data_in_leaf=20, bagging_fraction=0.8, bagging_freq=1,
    feature_fraction=0.9, seed=7, deterministic=True, num_threads=4, verbose=-1,
)
ROUNDS = 600


def fit(train: pd.DataFrame, quantile: float | None = None) -> lgb.Booster:
    objective = {"objective": "quantile", "alpha": quantile} if quantile else {"objective": "poisson"}
    data = lgb.Dataset(train[FEATURES], train["transactions"], categorical_feature=["outlet"])
    return lgb.train({**PARAMS, **objective}, data, num_boost_round=ROUNDS)


def predict(model: lgb.Booster, rows: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(rows[FEATURES]), 0, None)


def with_baselines(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(last_week_rule=rows["lag_7d"], four_week_avg=rows["mean_4w_same_hour"])


def backtest(frame: pd.DataFrame, history_end: pd.Timestamp, weeks: int = BACKTEST_WEEKS) -> pd.DataFrame:
    start = frame["timestamp"].min().normalize()
    available = ((history_end - start).days + 1 - MIN_TRAIN_DAYS) // HORIZON_DAYS
    if available < 1:
        raise SystemExit(f"Need at least {MIN_TRAIN_DAYS + HORIZON_DAYS} days of sales history to back-test; have {(history_end - start).days + 1}.")
    if available < weeks:
        log.warning("Only enough history for a %d-week back-test (wanted %d)", available, weeks)
        weeks = available

    folds = []
    for k in range(weeks, 0, -1):
        origin = history_end - pd.Timedelta(days=HORIZON_DAYS * k)  # last day the model is allowed to see
        cut = origin + pd.Timedelta(days=1)
        train = frame[frame["timestamp"] < cut]
        test = frame[(frame["timestamp"] >= cut) & (frame["timestamp"] < cut + pd.Timedelta(days=HORIZON_DAYS))]
        model = fit(train)
        folds.append(with_baselines(test).assign(origin=origin, forecast=predict(model, test)))
        log.info("Back-test fold from %s: trained on %d rows", f"{origin:%Y-%m-%d}", len(train))
    cols = ["timestamp", "outlet", "origin", "transactions", "forecast", "last_week_rule", "four_week_avg"]
    out = pd.concat(folds)[cols].rename(columns={"transactions": "actual"})
    out["outlet"] = out["outlet"].astype(str)
    return out.reset_index(drop=True)


def accuracy(bt: pd.DataFrame) -> pd.DataFrame:
    """1 - WAPE per outlet (and for all outlets together), per method, at hourly and daily level."""
    hourly = bt
    daily = bt.assign(date=bt["timestamp"].dt.normalize()).groupby(["outlet", "date"], as_index=False)[
        ["actual", *METHODS.values()]].sum(min_count=1)
    rows = []
    for level, df in (("hour", hourly), ("day", daily)):
        for outlet, g in [*df.groupby("outlet"), ("All outlets", df)]:
            for method, col in METHODS.items():
                ok = g[col].notna()
                wape = (g.loc[ok, "actual"] - g.loc[ok, col]).abs().sum() / g.loc[ok, "actual"].sum()
                rows.append({"level": level, "outlet": outlet, "method": method, "accuracy": round(1 - wape, 4)})
    return pd.DataFrame(rows)


def print_accuracy(acc: pd.DataFrame, weeks: int) -> None:
    for level in ("hour", "day"):
        table = acc[acc["level"] == level].pivot(index="outlet", columns="method", values="accuracy")[list(METHODS)]
        order = [o for o in table.index if o != "All outlets"] + ["All outlets"]
        print(f"\nBack-test accuracy (1 - WAPE), {'hourly' if level == 'hour' else 'daily'} totals, last {weeks} weeks, 7 days ahead:")
        print((table.loc[order] * 100).round(1).astype(str).add("%").to_string())


def forecast_window(frame: pd.DataFrame, history_end: pd.Timestamp) -> pd.DataFrame:
    cut = history_end + pd.Timedelta(days=1)
    train, future = frame[frame["timestamp"] < cut], frame[frame["timestamp"] >= cut]
    point = predict(fit(train), future)
    lower = predict(fit(train, quantile=0.1), future)
    upper = predict(fit(train, quantile=0.9), future)
    out = pd.DataFrame({
        "timestamp": future["timestamp"].to_numpy(),
        "outlet": future["outlet"].astype(str).to_numpy(),
        "forecast": point,
        "lower": np.minimum(lower, point),
        "upper": np.maximum(upper, point),
    })
    return out.round({"forecast": 1, "lower": 1, "upper": 1}).sort_values(["timestamp", "outlet"]).reset_index(drop=True)


def daily_sales(bt: pd.DataFrame, fc: pd.DataFrame, frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Back-test weeks (actual vs what the model would have said) followed by the forecast week."""
    past = bt.assign(date=bt["timestamp"].dt.normalize()).groupby(["date", "outlet"], as_index=False)[
        ["actual", "forecast", "last_week_rule"]].sum()
    future = fc.assign(date=fc["timestamp"].dt.normalize())
    lw = with_baselines(frame)[["timestamp", "outlet", "last_week_rule"]].assign(outlet=lambda d: d["outlet"].astype(str))
    future = future.merge(lw, on=["timestamp", "outlet"], how="left")
    future = future.groupby(["date", "outlet"], as_index=False)[["forecast", "last_week_rule"]].sum().assign(actual=np.nan)
    out = pd.concat([past, future]).merge(signals[["date", "rain_mm", "public_holiday"]], on="date", how="left")
    out[["actual", "forecast", "last_week_rule"]] = out[["actual", "forecast", "last_week_rule"]].round(0)
    cols = ["date", "outlet", "actual", "forecast", "last_week_rule", "rain_mm", "public_holiday"]
    return out[cols].sort_values(["date", "outlet"]).reset_index(drop=True)


def run(client: ClientConfig) -> pd.DataFrame:
    inputs = load_inputs(client.inputs_dir)
    signals = load_signals(client.outputs_dir)
    history_end = inputs.history_end
    frame = add_features(build_grid(inputs.sales, history_end, HORIZON_DAYS), signals)

    bt = backtest(frame, history_end)
    acc = accuracy(bt)
    print_accuracy(acc, bt["origin"].nunique())
    fc = forecast_window(frame, history_end)

    out = client.outputs_dir
    fc.to_csv(out / "forecast.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    bt.round({c: 1 for c in METHODS.values()}).to_csv(out / "backtest.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    acc.to_csv(out / "backtest_accuracy.csv", index=False)
    daily_sales(bt, fc, frame, signals).to_csv(out / "daily_sales.csv", index=False, date_format="%Y-%m-%d")
    print(f"\nForecast {fc['timestamp'].min():%a %d %b} to {fc['timestamp'].max():%a %d %b %Y}: "
          f"{fc['forecast'].sum():,.0f} transactions across {fc['outlet'].nunique()} outlets -> {out / 'forecast.csv'}")
    return fc


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Back-test and forecast the next 7 days, hourly, per outlet.")
    parser.add_argument("--client", default="demo")
    run(load_client(parser.parse_args().client))


if __name__ == "__main__":
    main()
