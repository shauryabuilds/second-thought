"""From forecast to plan: suggested orders, suggested staffing, roster gaps, savings estimate.

    Suggested order = forecast demand x (1 + buffer) - usable stock on hand,
                      rounded up to whole packs, never negative.
        forecast demand = forecast transactions over the item's cover days x units per
        transaction. Units per transaction are learned from items.csv (last 28 days,
        per outlet) when it exists, else taken from menu.csv.
    Suggested staff = ceil(forecast transactions / transactions per staff-hour),
                      at least the outlet's minimum while it is open.
    Gaps            = rostered vs needed staff-hours over the last 28 days, where
                      "needed" applies the same rule to the transactions that happened.
    Savings         = an ESTIMATE from the assumptions in client.yaml. Not a guarantee.

Writes to the client's outputs folder:
  staffing_plan.csv     date, outlet, hour, forecast_orders, suggested_staff, scheduled_staff
  suggested_orders.csv  item_key, outlet, item, unit, forecast_demand, on_hand, suggested_order
  outlet_summary.csv    outlet, waste_sgd, overstaffed_hours, understaffed_hours, stockouts
  staffing_gaps.csv     per outlet and day, rostered vs needed staff-hours (last 28 days)
  item_rates.csv        units per transaction used for each outlet and item, and where it came from
  savings_estimate.csv  the estimate, with every assumption written next to it
"""

from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from kaizen.config import ClientConfig, load_client, setup_logging
from kaizen.contract import Inputs, load_inputs

log = logging.getLogger(__name__)

# Hours forecast below this many transactions are treated as closed (no staff suggested).
OPEN_THRESHOLD = 1.0


def staff_needed(transactions, transactions_per_staff_hour: float, min_staff: int,
                 open_threshold: float = OPEN_THRESHOLD) -> np.ndarray:
    t = np.asarray(transactions, dtype=float)
    need = np.maximum(min_staff, np.ceil(t / transactions_per_staff_hour - 1e-9))
    return np.where(t >= open_threshold, need, 0).astype(int)


def order_quantity(forecast_demand: float, buffer: float, on_hand: float, pack_size: float) -> float:
    shortfall = forecast_demand * (1 + buffer) - on_hand
    if shortfall <= 0:
        return 0.0
    return float(np.ceil(shortfall / pack_size - 1e-9) * pack_size)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def review_window(inputs: Inputs, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = inputs.history_end
    return end - pd.Timedelta(days=days - 1), end


def item_rates(inputs: Inputs, review_days: int) -> pd.DataFrame:
    """Units used per transaction, per outlet and item."""
    start, end = review_window(inputs, review_days)
    rows = []
    units = {}
    if inputs.stock is not None:
        units.update(inputs.stock.drop_duplicates("item", keep="last").set_index("item")["unit"].to_dict())
    if inputs.menu is not None:
        units.update(inputs.menu.set_index("item")["unit"].to_dict())

    if inputs.items is not None:
        recent = inputs.items[(inputs.items["date"] >= start) & (inputs.items["date"] <= end)]
        sales = inputs.sales[inputs.sales["timestamp"].dt.normalize() >= start]
        txns = sales.groupby("outlet")["transactions"].sum()
        for (outlet, item), units_sold in recent.groupby(["outlet", "item"])["units_sold"].sum().items():
            rows.append({"outlet": outlet, "item": item, "unit": units.get(item, ""),
                         "usage_per_transaction": units_sold / txns[outlet], "source": f"items.csv, last {review_days} days"})
    if inputs.menu is not None:
        have = {(r["outlet"], r["item"]) for r in rows}
        for outlet in inputs.outlets:
            for m in inputs.menu.itertuples():
                if (outlet, m.item) not in have:
                    rows.append({"outlet": outlet, "item": m.item, "unit": m.unit,
                                 "usage_per_transaction": m.usage_per_transaction, "source": "menu.csv"})
    return pd.DataFrame(rows, columns=["outlet", "item", "unit", "usage_per_transaction", "source"])


def suggested_orders(client: ClientConfig, inputs: Inputs, forecast: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    cols = ["item_key", "outlet", "item", "unit", "forecast_demand", "on_hand", "suggested_order"]
    if rates.empty:
        log.warning("No items.csv or menu.csv: no suggested orders")
        return pd.DataFrame(columns=cols)
    day = forecast.assign(date=forecast["timestamp"].dt.normalize())
    daily = day.groupby(["outlet", "date"])["forecast"].sum()
    first_day = day["date"].min()
    latest_stock = {}
    if inputs.stock is not None:
        last = inputs.stock[inputs.stock["date"] == inputs.stock["date"].max()]
        latest_stock = last.set_index(["outlet", "item"])["on_hand"].to_dict()

    rows = []
    for r in rates.itertuples():
        p = client.item_params(r.item)
        window = daily.loc[r.outlet]
        txns = window[window.index < first_day + pd.Timedelta(days=p.cover_days)].sum()
        demand = txns * r.usage_per_transaction
        on_hand = 0.0 if p.perishable else latest_stock.get((r.outlet, r.item), 0.0)
        rows.append({
            "item_key": f"{slug(r.outlet)}.{slug(r.item)}", "outlet": r.outlet, "item": r.item, "unit": r.unit,
            "forecast_demand": round(demand, 1), "on_hand": round(on_hand, 1),
            "suggested_order": order_quantity(demand, p.buffer, on_hand, p.pack_size),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(["outlet", "item"]).reset_index(drop=True)


def staffing_plan(client: ClientConfig, inputs: Inputs, forecast: pd.DataFrame) -> pd.DataFrame:
    df = forecast.assign(date=forecast["timestamp"].dt.normalize(), hour=forecast["timestamp"].dt.hour)
    df["suggested_staff"] = 0
    for outlet, idx in df.groupby("outlet").groups.items():
        p = client.staffing_for(outlet)
        df.loc[idx, "suggested_staff"] = staff_needed(df.loc[idx, "forecast"], p.transactions_per_staff_hour, p.min_staff)
    if inputs.roster is not None:
        df = df.merge(inputs.roster, on=["date", "outlet", "hour"], how="left")
        df["scheduled_staff"] = df["scheduled_staff"].fillna(0).astype(int)
    else:
        df["scheduled_staff"] = pd.NA
    df = df.rename(columns={"forecast": "forecast_orders"})
    cols = ["date", "outlet", "hour", "forecast_orders", "suggested_staff", "scheduled_staff"]
    return df[cols].sort_values(["date", "outlet", "hour"]).reset_index(drop=True)


def staffing_gaps(client: ClientConfig, inputs: Inputs) -> pd.DataFrame:
    """Per outlet and day over the review window: rostered vs needed staff-hours, with hindsight."""
    cols = ["date", "outlet", "rostered_staff_hours", "needed_staff_hours", "overstaffed_hours", "understaffed_hours"]
    if inputs.roster is None:
        return pd.DataFrame(columns=cols)
    start, end = review_window(inputs, client.review_days)
    sales = inputs.sales.assign(date=inputs.sales["timestamp"].dt.normalize(), hour=inputs.sales["timestamp"].dt.hour)
    sales = sales[(sales["date"] >= start) & (sales["date"] <= end)]
    roster = inputs.roster[(inputs.roster["date"] >= start) & (inputs.roster["date"] <= end)]
    df = sales.merge(roster, on=["date", "outlet", "hour"], how="outer")
    df[["transactions", "scheduled_staff"]] = df[["transactions", "scheduled_staff"]].fillna(0)
    df["needed"] = 0
    for outlet, idx in df.groupby("outlet").groups.items():
        p = client.staffing_for(outlet)
        df.loc[idx, "needed"] = staff_needed(df.loc[idx, "transactions"], p.transactions_per_staff_hour, p.min_staff)
    df["over"] = (df["scheduled_staff"] - df["needed"]).clip(lower=0)
    df["under"] = (df["needed"] - df["scheduled_staff"]).clip(lower=0)
    out = df.groupby(["date", "outlet"], as_index=False).agg(
        rostered_staff_hours=("scheduled_staff", "sum"), needed_staff_hours=("needed", "sum"),
        overstaffed_hours=("over", "sum"), understaffed_hours=("under", "sum"))
    return out[cols].astype({c: int for c in cols[2:]})


def outlet_summary(client: ClientConfig, inputs: Inputs, gaps: pd.DataFrame) -> pd.DataFrame:
    """Last review_days days: waste, staffing gaps, and item-days that closed with zero stock."""
    start, end = review_window(inputs, client.review_days)
    out = pd.DataFrame({"outlet": inputs.outlets})

    def recent(df: pd.DataFrame | None) -> pd.DataFrame | None:
        return None if df is None else df[(df["date"] >= start) & (df["date"] <= end)]

    waste = recent(inputs.waste)
    out["waste_sgd"] = out["outlet"].map(waste.groupby("outlet")["waste_sgd"].sum()).round(2) if waste is not None else np.nan
    g = gaps.groupby("outlet")[["overstaffed_hours", "understaffed_hours"]].sum()
    for col in ("overstaffed_hours", "understaffed_hours"):
        out[col] = out["outlet"].map(g[col]) if not gaps.empty else np.nan
    stock = recent(inputs.stock)
    out["stockouts"] = out["outlet"].map(stock[stock["on_hand"] <= 0].groupby("outlet").size()).fillna(0).astype(int) if stock is not None else np.nan
    return out


def savings_estimate(client: ClientConfig, summary: pd.DataFrame) -> pd.DataFrame:
    a = client.savings
    to_month = 30.4 / client.review_days
    df = summary[["outlet", "waste_sgd", "overstaffed_hours"]].copy()
    df = pd.concat([df, pd.DataFrame([{"outlet": "All outlets", "waste_sgd": df["waste_sgd"].sum(min_count=1),
                                        "overstaffed_hours": df["overstaffed_hours"].sum(min_count=1)}])], ignore_index=True)
    df["avoidable_waste_sgd_per_month"] = (df["waste_sgd"] * a.avoidable_waste_share * to_month).round(0)
    df["recoverable_wages_sgd_per_month"] = (df["overstaffed_hours"] * a.recoverable_overstaff_share * a.hourly_wage_sgd * to_month).round(0)
    df["estimated_saving_sgd_per_month"] = df[["avoidable_waste_sgd_per_month", "recoverable_wages_sgd_per_month"]].sum(axis=1, min_count=1)
    df = df.rename(columns={"waste_sgd": f"waste_sgd_last_{client.review_days}d",
                            "overstaffed_hours": f"overstaffed_hours_last_{client.review_days}d"})
    return df.assign(
        hourly_wage_sgd=a.hourly_wage_sgd, avoidable_waste_share=a.avoidable_waste_share,
        recoverable_overstaff_share=a.recoverable_overstaff_share,
        basis="ESTIMATE from the assumptions in this row; not a guarantee",
    )


def run(client: ClientConfig) -> None:
    inputs = load_inputs(client.inputs_dir)
    out = client.outputs_dir
    path = out / "forecast.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make forecast` first.")
    forecast = pd.read_csv(path, parse_dates=["timestamp"])

    rates = item_rates(inputs, client.review_days)
    orders = suggested_orders(client, inputs, forecast, rates)
    staffing = staffing_plan(client, inputs, forecast)
    gaps = staffing_gaps(client, inputs)
    summary = outlet_summary(client, inputs, gaps)
    savings = savings_estimate(client, summary)

    rates.round({"usage_per_transaction": 4}).to_csv(out / "item_rates.csv", index=False)
    orders.to_csv(out / "suggested_orders.csv", index=False)
    staffing.to_csv(out / "staffing_plan.csv", index=False, date_format="%Y-%m-%d")
    gaps.to_csv(out / "staffing_gaps.csv", index=False, date_format="%Y-%m-%d")
    summary.to_csv(out / "outlet_summary.csv", index=False)
    savings.to_csv(out / "savings_estimate.csv", index=False)

    print(f"Plan for {client.name}, week of {forecast['timestamp'].min():%d %b %Y} -> {out}")
    print(f"  suggested orders: {int((orders['suggested_order'] > 0).sum())} lines across {orders['outlet'].nunique()} outlets")
    print(f"  suggested staff-hours next 7 days: {staffing['suggested_staff'].sum():,}"
          + (f" (rostered: {staffing['scheduled_staff'].sum():,})" if inputs.roster is not None else ""))
    print(f"\nLast {client.review_days} days by outlet:\n{summary.to_string(index=False)}")
    total = savings.iloc[-1]["estimated_saving_sgd_per_month"]
    if pd.notna(total):
        print(f"\nEstimated saving (an estimate from client.yaml assumptions, not a guarantee): about S${total:,.0f} per month")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Turn the forecast into orders, staffing and a savings estimate.")
    parser.add_argument("--client", default="demo")
    run(load_client(parser.parse_args().client))


if __name__ == "__main__":
    main()
