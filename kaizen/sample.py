"""Sample data for "Demo Café Group", a fictional five-outlet café chain.

Writes 52 weeks of history plus a 7-day forecast window into data/sample/, in
exactly the format of the data contract (kaizen/contract.py), so the rest of
the pipeline cannot tell it apart from a real client's files.

How the numbers are made
  Daily transactions per outlet = base level x weekday pattern x public holiday
  x school holiday x season x rain x slow growth x day-to-day noise. Each day is
  then spread over opening hours with an outlet-specific hourly shape, and each
  hour gets count noise.
  Weather uses real MSS records where available (so a rainy day in the demo
  really was rainy) and a synthetic Singapore-like series elsewhere.
  Stock, waste and rosters are simulated with the rules of thumb operators use
  today (fixed weekly par levels, "last week plus 15%" for fresh items, one
  roster template per day of week), which is where the gaps in the demo come from.

It also writes two messy POS exports to data/clients/example_*/raw/ for
trying out `make ingest`.

No customer data of any kind is generated.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kaizen import signals
from kaizen.config import HORIZON_DAYS, ROOT, SIGNALS_DIR, setup_logging

log = logging.getLogger(__name__)

SAMPLE_DIR = ROOT / "data" / "sample"
HISTORY_DAYS = 364  # 52 weeks, so the history starts on the same weekday it ends + 1


@dataclass(frozen=True)
class Outlet:
    name: str
    code: str
    base: float  # typical Monday-Thursday transactions per day
    weekday: tuple[float, ...]  # Mon..Sun multipliers
    public_holiday: float  # multiplier used instead of the weekday one
    school_holiday: float
    rain_max_drop: float  # share of trade lost on a very wet day
    hours_weekday: tuple[int, int]  # first hour open, first hour closed
    hours_weekend: tuple[int, int]
    shape_weekday: tuple[tuple[float, float, float], ...]  # (peak hour, width, weight) bumps
    shape_weekend: tuple[tuple[float, float, float], ...]
    month: tuple[float, ...] = (1,) * 12  # Jan..Dec season multipliers


OUTLETS = [
    Outlet(
        "CBD - Raffles Place", "RP01", 1000, (1.0, 1.06, 1.06, 1.0, 0.78, 0.28, 0.20), 0.22, 0.94, 0.06,
        (7, 20), (8, 17),
        ((8.3, 0.8, 1.7), (12.5, 0.8, 1.9), (15.5, 1.5, 0.5)),
        ((10.5, 1.5, 1.0), (13.0, 1.2, 1.0)),
        month=(0.92, 1.0, 1.0, 1.0, 1.0, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86),
    ),
    Outlet(
        "Orchard", "OR02", 640, (1.0, 0.97, 1.0, 1.03, 1.15, 1.55, 1.45), 1.45, 1.08, 0.28,
        (9, 22), (9, 22),
        ((11.0, 1.2, 0.5), (15.0, 1.8, 1.1), (19.0, 1.5, 0.9)),
        ((12.0, 1.5, 0.8), (15.5, 2.0, 1.2), (19.0, 1.5, 0.9)),
        month=(1.0, 1.05, 0.98, 0.98, 1.0, 1.02, 1.0, 0.98, 0.97, 0.98, 1.08, 1.22),
    ),
    Outlet(
        "Tampines", "TP03", 600, (1.0, 0.98, 1.0, 1.0, 1.06, 1.42, 1.38), 1.35, 1.18, 0.08,
        (8, 22), (8, 22),
        ((9.0, 1.0, 0.7), (13.0, 1.5, 0.9), (16.0, 1.5, 0.8), (19.0, 1.3, 0.8)),
        ((10.0, 1.5, 1.0), (14.0, 1.8, 1.0), (19.0, 1.5, 0.8)),
    ),
    Outlet(
        "Jurong East", "JE04", 560, (1.0, 1.0, 0.98, 1.0, 1.05, 1.40, 1.36), 1.32, 1.16, 0.08,
        (8, 22), (8, 22),
        ((9.0, 1.0, 0.6), (13.0, 1.5, 1.0), (16.0, 1.5, 0.8), (19.5, 1.3, 0.9)),
        ((10.0, 1.5, 1.0), (14.0, 1.8, 1.0), (19.5, 1.5, 0.8)),
    ),
    Outlet(
        "Changi Airport T3", "CG05", 820, (1.0, 0.97, 0.97, 1.0, 1.05, 1.06, 1.04), 1.05, 1.10, 0.0,
        (5, 24), (5, 24),
        ((7.0, 1.5, 1.2), (12.5, 2.0, 0.9), (19.0, 2.0, 1.0)),
        ((7.0, 1.5, 1.2), (12.5, 2.0, 0.9), (19.0, 2.0, 1.0)),
        month=(1.06, 1.08, 1.02, 0.96, 0.96, 1.16, 1.10, 1.0, 0.96, 0.97, 1.04, 1.22),
    ),
]

# item, unit, usage per transaction (group average), per-outlet usage multipliers, unit cost S$
MENU = [
    ("Coffee beans", "kg", 0.015, {"RP01": 1.15, "OR02": 0.9, "CG05": 0.95}, 38.0),
    ("Fresh milk", "L", 0.11, {"RP01": 1.1, "OR02": 0.95}, 2.4),
    ("Butter croissant", "pcs", 0.20, {"RP01": 0.9, "OR02": 1.3, "CG05": 1.1}, 1.2),
    ("Chicken sandwich", "pcs", 0.10, {"RP01": 1.5, "CG05": 1.3, "OR02": 0.8}, 2.8),
    ("Takeaway cups", "pcs", 0.55, {"RP01": 1.35, "CG05": 1.4, "OR02": 0.7, "TP03": 0.8, "JE04": 0.8}, 0.12),
]
FRESH_ITEMS = {"Butter croissant", "Chicken sandwich"}
# Dry goods: delivery weekdays (0 = Monday) under the operator's current fixed-par routine.
DELIVERY_DAYS = {"Coffee beans": (0,), "Fresh milk": (0, 3), "Takeaway cups": (0,)}
OPERATOR_TXN_PER_STAFF_HOUR = 20

# Singapore climate, by month: chance of a rain day, monthly rainfall mm, mean daily max temp °C.
RAIN_CHANCE = (0.50, 0.35, 0.45, 0.50, 0.45, 0.40, 0.42, 0.45, 0.45, 0.50, 0.60, 0.60)
RAIN_MM = (240, 110, 180, 165, 170, 130, 155, 160, 150, 170, 255, 310)
MAX_TEMP = (30.6, 31.6, 32.2, 32.4, 32.3, 31.8, 31.3, 31.3, 31.4, 31.6, 31.0, 30.3)


def last_sunday(today: pd.Timestamp | None = None) -> pd.Timestamp:
    today = (today or pd.Timestamp.today()).normalize()
    return today - pd.Timedelta(days=(today.weekday() + 1) % 7 or 7)


def synthetic_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    m = dates.month.to_numpy() - 1
    chance = np.take(RAIN_CHANCE, m)
    wet_day_mean = np.take(RAIN_MM, m) / (30.4 * chance)
    rain = np.where(rng.random(len(dates)) < chance, rng.gamma(0.7, wet_day_mean / 0.7), 0.0)
    temp = np.take(MAX_TEMP, m) - 0.05 * np.minimum(rain, 40) + rng.normal(0, 0.7, len(dates))
    return pd.DataFrame({"date": dates, "rain_mm": rain.round(1), "max_temp_c": temp.round(1)})


def hourly_shape(bumps: tuple[tuple[float, float, float], ...], hours: tuple[int, int]) -> pd.Series:
    h = np.arange(*hours)
    w = 0.15 + sum(weight * np.exp(-0.5 * ((h + 0.5 - peak) / width) ** 2) for peak, width, weight in bumps)
    return pd.Series(w / w.sum(), index=h)


def daily_expected(outlet: Outlet, days: pd.DataFrame, noise: bool, rng: np.random.Generator) -> np.ndarray:
    """Expected transactions per day. With noise=False and no calendar effects this is the 'typical' day."""
    dow = days["date"].dt.weekday.to_numpy()
    f = np.take(outlet.weekday, dow)
    f = np.where(days["is_public_holiday"] == 1, outlet.public_holiday, f)
    f = f * np.where(days["is_school_holiday"] == 1, outlet.school_holiday, 1.0)
    if outlet.code == "RP01":  # office workers bridge public holidays with leave
        near_ph = (days["days_to_ph"] == 1) | (days["days_from_ph"] == 1)
        f = f * np.where(near_ph & (dow < 5) & (days["is_public_holiday"] == 0), 0.82, 1.0)
    f = f * np.take(outlet.month, days["date"].dt.month.to_numpy() - 1)
    f = f * (1 - outlet.rain_max_drop * (1 - np.exp(-days["rain_mm"].to_numpy() / 15)))
    f = f * (1 + 0.05 * days["t"].to_numpy() / 365)
    if noise:
        f = f * rng.lognormal(0, 0.05, len(days))
    return outlet.base * f


def is_weekend_like(days: pd.DataFrame) -> np.ndarray:
    return ((days["date"].dt.weekday >= 5) | (days["is_public_holiday"] == 1)).to_numpy()


def make_sales(days: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for outlet in OUTLETS:
        expected = daily_expected(outlet, days, noise=True, rng=rng)
        weekend_like = is_weekend_like(days)
        for kind, hours, bumps in (("wd", outlet.hours_weekday, outlet.shape_weekday),
                                   ("we", outlet.hours_weekend, outlet.shape_weekend)):
            mask = weekend_like if kind == "we" else ~weekend_like
            shape = hourly_shape(bumps, hours)
            lam = np.outer(expected[mask], shape.to_numpy())
            lam = lam * rng.gamma(25, 1 / 25, lam.shape)  # extra hour-to-hour variation
            counts = rng.poisson(lam)
            sel = days.loc[mask, "date"].to_numpy()
            frames.append(pd.DataFrame({
                "timestamp": (np.repeat(sel, len(shape)) + np.tile(shape.index.to_numpy() * 3600, len(sel)).astype("timedelta64[s]")),
                "outlet": outlet.name,
                "transactions": counts.ravel(),
            }))
    return pd.concat(frames).sort_values(["timestamp", "outlet"]).reset_index(drop=True)


def make_roster(days: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One fixed template per outlet and weekday, built from a typical week.

    That is how most rosters are made today, and it ignores holidays, school
    breaks and weather. Public holidays on weekdays get the weekday template.
    """
    typical = days.assign(is_public_holiday=0, is_school_holiday=0, days_to_ph=99, days_from_ph=99, rain_mm=0.0, t=180)
    frames = []
    for outlet in OUTLETS:
        base_day = daily_expected(outlet, typical, noise=False, rng=rng)
        for dow in range(7):
            mask = (days["date"].dt.weekday == dow).to_numpy()
            hours, bumps = (outlet.hours_weekend, outlet.shape_weekend) if dow >= 5 else (outlet.hours_weekday, outlet.shape_weekday)
            shape = hourly_shape(bumps, hours)
            staff = np.maximum(2, np.ceil(base_day[mask][0] * shape.to_numpy() * 1.05 / OPERATOR_TXN_PER_STAFF_HOUR))
            sel = days.loc[mask, "date"].to_numpy()
            grid = np.tile(staff, (len(sel), 1))
            grid = grid + rng.choice([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], size=grid.shape)
            frames.append(pd.DataFrame({
                "date": np.repeat(sel, len(shape)),
                "outlet": outlet.name,
                "hour": np.tile(shape.index.to_numpy(), len(sel)),
                "scheduled_staff": np.maximum(1, grid).astype(int).ravel(),
            }))
    return pd.concat(frames).sort_values(["date", "outlet", "hour"]).reset_index(drop=True)


def make_items_stock_waste(sales: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = sales.assign(date=sales["timestamp"].dt.normalize()).groupby(["outlet", "date"])["transactions"].sum()
    items_rows, stock_rows, waste_rows = [], [], []
    for outlet in OUTLETS:
        txns = daily.loc[outlet.name]
        dates = txns.index
        waste = np.zeros(len(dates))
        for item, unit, rate, mult, cost in MENU:
            use_rate = rate * mult.get(outlet.code, 1.0)
            if item in FRESH_ITEMS:
                demand = rng.poisson(txns.to_numpy() * use_rate)
                sold = np.zeros(len(dates))
                for i in range(len(dates)):
                    # Operator's rule: same as last week, plus 15%.
                    ref = sold[i - 7] if i >= 7 else txns.iloc[i] * use_rate
                    delivered = round(ref * 1.15)
                    sold[i] = min(demand[i], delivered)
                    waste[i] += (delivered - sold[i]) * cost
                units = sold
            else:
                units = txns.to_numpy() * use_rate * rng.normal(1, 0.04, len(dates))
                deliveries = DELIVERY_DAYS[item]
                cover = max(np.diff([*deliveries, deliveries[0] + 7]))  # longest gap between deliveries
                par = units.mean() * cover * 1.12  # set once, never revisited
                on_hand = par
                for i, d in enumerate(dates):
                    if d.weekday() in deliveries:
                        on_hand = max(on_hand, par)  # top up to par
                    on_hand = max(0.0, on_hand - units[i])
                    if unit == "pcs":
                        on_hand = float(np.floor(on_hand))
                    stock_rows.append((d, outlet.name, item, unit, round(on_hand, 2)))
                if item == "Fresh milk":
                    waste += rng.gamma(1.5, 2.0, len(dates))  # the odd spoiled carton
            decimals = 0 if unit == "pcs" else 2
            items_rows += [(d, outlet.name, item, round(u, decimals)) for d, u in zip(dates, units)]
        waste_rows += [(d, outlet.name, round(w, 2)) for d, w in zip(dates, waste)]
    items = pd.DataFrame(items_rows, columns=["date", "outlet", "item", "units_sold"])
    stock = pd.DataFrame(stock_rows, columns=["date", "outlet", "item", "unit", "on_hand"])
    waste = pd.DataFrame(waste_rows, columns=["date", "outlet", "waste_sgd"])
    return items, stock, waste


def calendar(dates: pd.DatetimeIndex, weather: pd.DataFrame, signals_dir: Path, allow_network: bool) -> pd.DataFrame:
    ph = signals.public_holidays(dates[0] - pd.Timedelta(days=30), dates[-1] + pd.Timedelta(days=30), signals_dir, allow_network)
    school = signals.school_holiday_dates(dates[0], dates[-1], signals_dir)
    days = pd.DataFrame({"date": dates}).merge(weather, on="date")
    days["is_public_holiday"] = days["date"].isin(ph["date"]).astype(int)
    days["is_school_holiday"] = days["date"].isin(school).astype(int)
    ph_days = ph["date"].to_numpy()
    diff = (ph_days[None, :] - days["date"].to_numpy()[:, None]) / np.timedelta64(1, "D")
    days["days_to_ph"] = np.where(diff >= 0, diff, np.inf).min(axis=1)
    days["days_from_ph"] = np.where(diff <= 0, -diff, np.inf).min(axis=1)
    days["t"] = np.arange(len(days))
    return days


def write_messy_exports(sales: pd.DataFrame, stock: pd.DataFrame, waste: pd.DataFrame, roster: pd.DataFrame,
                        rng: np.random.Generator) -> None:
    """Two fake 'real world' exports for trying `make ingest`."""
    branch = {o.name: f"DEMO CAFE {o.code}" for o in OUTLETS}
    code = {o.name: o.code for o in OUTLETS}

    # example_15min: a bucketed sales report, plus stock, waste and roster with their own column names.
    raw = ROOT / "data" / "clients" / "example_15min" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    quarters = rng.multinomial(sales["transactions"].to_numpy(), [0.25] * 4)
    q = pd.DataFrame({
        "Interval": (np.repeat(sales["timestamp"].to_numpy(), 4) + np.tile(np.arange(4) * 15 * 60, len(sales)).astype("timedelta64[s]")),
        "Branch Name": np.repeat(sales["outlet"].map(branch).to_numpy(), 4),
        "No. of Bills": quarters.ravel(),
    })
    q["Gross Sales (S$)"] = (q["No. of Bills"] * rng.normal(7.2, 0.6, len(q))).round(2)
    q["Interval"] = pd.to_datetime(q["Interval"]).dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
    q.to_csv(raw / "sales_15min_export.csv", index=False)
    pd.DataFrame({
        "Count Date": stock["date"].dt.strftime("%d-%b-%Y"), "Branch Name": stock["outlet"].map(branch),
        "SKU": stock["item"], "UOM": stock["unit"], "Closing Qty": stock["on_hand"],
    }).to_csv(raw / "inventory_counts.csv", index=False)
    pd.DataFrame({"Date": waste["date"].dt.strftime("%d/%m/%Y"), "Branch Name": waste["outlet"].map(branch),
                  "Wastage Value (S$)": waste["waste_sgd"]}).to_csv(raw / "wastage_log.csv", index=False)
    pd.DataFrame({"Shift Date": roster["date"].dt.strftime("%Y-%m-%d"), "Branch Name": roster["outlet"].map(branch),
                  "Hour": roster["hour"], "Staff On Duty": roster["scheduled_staff"]}).to_csv(raw / "roster_export.csv", index=False)
    pd.DataFrame({"Item": [m[0] for m in MENU], "UOM": [m[1] for m in MENU],
                  "Qty per Bill": [m[2] for m in MENU]}).to_csv(raw / "recipe_usage.csv", index=False)

    # example_txn: line-item level receipts for the last 4 weeks, including voids. No customer fields.
    raw = ROOT / "data" / "clients" / "example_txn" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    recent = sales[sales["timestamp"] > sales["timestamp"].max() - pd.Timedelta(days=28)]
    n = recent["transactions"].to_numpy()
    n_void = rng.binomial(n, 0.015)
    ts = np.repeat(recent["timestamp"].to_numpy(), n + n_void) + rng.integers(0, 3600, (n + n_void).sum()).astype("timedelta64[s]")
    outlet = np.repeat(recent["outlet"].to_numpy(), n + n_void)
    status = np.concatenate([np.r_[np.full(a, "COMPLETED"), np.full(b, "VOID")] for a, b in zip(n, n_void)])
    receipts = pd.DataFrame({"ts": ts, "outlet": outlet, "Status": status}).sort_values("ts", kind="stable")
    receipts["Receipt No"] = receipts["outlet"].map(code) + "-" + (receipts.groupby("outlet").cumcount() + 100001).astype(str)
    lines = receipts.loc[receipts.index.repeat(rng.choice([1, 1, 2, 2, 3], len(receipts)))]
    menu_lines = ["Flat White", "Long Black", "Iced Latte", "Butter Croissant", "Chicken Sandwich", "Banana Bread"]
    pd.DataFrame({
        "Receipt No": lines["Receipt No"],
        "Txn Date": lines["ts"].dt.strftime("%d/%m/%Y"),
        "Txn Time": lines["ts"].dt.strftime("%H:%M:%S"),
        "Store": lines["outlet"].map(code),
        "Line Item": rng.choice(menu_lines, len(lines)),
        "Net Amount": rng.choice([4.5, 5.2, 5.8, 3.9, 7.5, 4.2], len(lines)),
        "Payment Type": rng.choice(["Card", "PayNow", "Cash", "NETS"], len(lines)),
        "Status": lines["Status"],
    }).to_csv(raw / "pos_transactions_export.csv", index=False)
    log.info("Messy example exports written to data/clients/example_15min/raw and data/clients/example_txn/raw")


def generate(end: pd.Timestamp, seed: int = 7, out_dir: Path = SAMPLE_DIR, signals_dir: Path = SIGNALS_DIR,
             allow_network: bool = True, messy_exports: bool = True) -> None:
    rng = np.random.default_rng(seed)
    start = end - pd.Timedelta(days=HISTORY_DAYS - 1)
    window_end = end + pd.Timedelta(days=HORIZON_DAYS)
    all_days = pd.date_range(start, window_end, freq="D")
    history = all_days[all_days <= end]

    # Real MSS weather where it exists, synthetic for the rest (and for the forecast week).
    synth = synthetic_weather(all_days, rng)
    weather = signals.weather(start, window_end, fallback=synth, fallback_label="synthetic",
                              signals_dir=signals_dir, allow_network=allow_network)
    weather = weather[["date", "rain_mm", "max_temp_c"]]

    days_all = calendar(all_days, weather, signals_dir, allow_network)
    days = days_all[days_all["date"] <= end].reset_index(drop=True)
    sales = make_sales(days, rng)
    roster = make_roster(days_all, rng)
    items, stock, waste = make_items_stock_waste(sales, rng)
    menu = pd.DataFrame([(m[0], m[1], m[2]) for m in MENU], columns=["item", "unit", "usage_per_transaction"])

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"sales": sales, "items": items, "stock": stock, "roster": roster, "waste": waste, "menu": menu, "weather": weather}
    for name, df in files.items():
        fmt = "%Y-%m-%d %H:%M:%S" if name == "sales" else "%Y-%m-%d"
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format=fmt)
    print(f"Sample data for Demo Café Group: history {history[0]:%a %d %b %Y} to {history[-1]:%a %d %b %Y}, "
          f"forecast week {end + pd.Timedelta(days=1):%d %b} to {window_end:%d %b %Y} -> {out_dir}")
    for name, df in files.items():
        print(f"  {name}.csv{'':<8}"[:16] + f"{len(df):>8,} rows")
    if messy_exports:
        write_messy_exports(sales, stock, waste, roster, rng)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate the Demo Café Group sample data.")
    parser.add_argument("--end", help="last day of history (YYYY-MM-DD). Default: last Sunday.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--offline", action="store_true", help="use cached signals only; synthetic weather where uncached")
    args = parser.parse_args()
    end = pd.Timestamp(args.end) if args.end else last_sunday()
    generate(end, seed=args.seed, allow_network=not args.offline)


if __name__ == "__main__":
    main()
