import numpy as np
import pandas as pd
import pytest

from kaizen.features import FEATURES, add_features, build_grid, holiday_distances
from kaizen.forecast import accuracy


def toy_inputs(days: int = 70):
    rng = np.random.default_rng(0)
    ts = [pd.Timestamp("2026-01-05") + pd.Timedelta(days=d, hours=h) for d in range(days) for h in range(8, 12)]
    sales = pd.DataFrame({"timestamp": ts, "outlet": "A", "transactions": rng.poisson(30, len(ts))})
    dates = pd.date_range("2026-01-05", periods=days + 21, freq="D")
    signals = pd.DataFrame({"date": dates, "is_public_holiday": 0, "is_school_holiday": 0, "rain_mm": 0.0, "max_temp_c": 31.0})
    signals.loc[signals["date"] == "2026-02-17", "is_public_holiday"] = 1
    return sales, signals


def test_grid_fills_closed_history_hours_with_zero_and_leaves_future_empty():
    sales, _ = toy_inputs(14)
    sales = sales.drop(index=5)  # an hour with no sales row
    grid = build_grid(sales, sales["timestamp"].max().normalize(), horizon_days=7)
    history = grid[grid["timestamp"] < "2026-01-19"]
    assert history["transactions"].notna().all() and (history["transactions"] == 0).sum() == 1
    assert grid[grid["timestamp"] >= "2026-01-19"]["transactions"].isna().all()


def test_features_never_use_the_last_7_days():
    """A 7-day-ahead forecast made at the end of day D knows nothing after D.

    So changing any sales after the end of day (target - 7) must not change the target's features.
    """
    sales, signals = toy_inputs()
    end = sales["timestamp"].max().normalize()
    base = add_features(build_grid(sales, end, 7), signals)
    target = pd.Timestamp("2026-03-01 09:00")
    changed = sales.copy()
    recent = changed["timestamp"] >= target.normalize() - pd.Timedelta(days=6)
    changed.loc[recent, "transactions"] = 9999
    other = add_features(build_grid(changed, end, 7), signals)
    a = base.loc[base["timestamp"] == target, FEATURES].iloc[0]
    b = other.loc[other["timestamp"] == target, FEATURES].iloc[0]
    pd.testing.assert_series_equal(a, b)


def test_lag_features_line_up():
    sales, signals = toy_inputs()
    df = add_features(build_grid(sales, sales["timestamp"].max().normalize(), 7), signals).set_index("timestamp")
    t = pd.Timestamp("2026-02-20 10:00")
    for k in (7, 14, 28):
        assert df.loc[t, f"lag_{k}d"] == df.loc[t - pd.Timedelta(days=k), "transactions"]


def test_holiday_distances():
    dates = pd.Series(pd.to_datetime(["2026-02-15", "2026-02-17", "2026-02-20", "2026-04-01"]))
    to_ph, from_ph = holiday_distances(dates, pd.Series(pd.to_datetime(["2026-02-17"])))
    assert to_ph.tolist() == [2, 0, 14, 14]
    assert from_ph.tolist() == [14, 0, 3, 14]


def test_accuracy_is_one_minus_wape():
    bt = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 09:00"]), "outlet": "A",
        "actual": [100.0, 100.0], "forecast": [90.0, 110.0], "last_week_rule": [50.0, 100.0], "four_week_avg": [100.0, 100.0],
    })
    acc = accuracy(bt).set_index(["level", "outlet", "method"])["accuracy"]
    assert acc[("hour", "A", "model")] == pytest.approx(0.9)          # 20 / 200 off
    assert acc[("day", "A", "model")] == pytest.approx(1.0)           # errors cancel within the day
    assert acc[("hour", "A", "last_week_rule")] == pytest.approx(0.75)
    assert acc[("hour", "All outlets", "four_week_avg")] == pytest.approx(1.0)
