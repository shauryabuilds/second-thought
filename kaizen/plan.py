"""From forecast to next week's plan: shifts, menu items, ingredients, orders.

    Shifts        Each day has three shifts (default 06-12, 12-18, 18-24; set in client.yaml).
                  People on a shift = enough for its busiest hour:
                  ceil(busiest-hour forecast transactions / transactions per staff-hour),
                  kept between min_staff and max_staff (3 to 6). A shift with no trade is closed.
                  Shifts that would need more than the maximum are flagged.
    Menu items    Forecast units of each menu item (SKU) per day = forecast transactions x that
                  item's share of transactions at that outlet over the last 28 days, worked out
                  separately for weekdays and for weekends/public holidays (items.csv).
                  Without items.csv, menu.csv gives the units per transaction.
    Ingredients   Menu items x recipes.csv = forecast use of each ingredient per day.
                  Without recipes.csv, each menu item is its own stock item.
    Orders        Forecast ingredient use over its cover days x (1 + buffer) - usable stock on
                  hand, rounded up to whole packs, never negative.
    Gaps          Over the last 28 days: people rostered on each shift vs the people the actual
                  transactions needed, in staff-hours.
    Savings       An ESTIMATE from the assumptions in client.yaml. Not a guarantee.

Writes to the client's outputs folder:
  staffing_plan.csv       date, outlet, shift, start, end, forecast_orders, busiest_hour_orders,
                          suggested_staff, needed_at_peak, scheduled_staff
  sku_forecast.csv        date, outlet, item, forecast_units
  ingredient_forecast.csv date, outlet, ingredient, unit, forecast_qty
  suggested_orders.csv    item_key, outlet, item, unit, forecast_demand, on_hand, suggested_order
  outlet_summary.csv      outlet, waste_sgd, overstaffed_hours, understaffed_hours, stockouts
  staffing_gaps.csv       per outlet, day and shift over the last 28 days
  sku_mix.csv             units per transaction used for each outlet, item and day type
  savings_estimate.csv    the estimate, with every assumption written next to it
"""

from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from kaizen.config import ClientConfig, load_client, setup_logging
from kaizen.contract import Inputs, load_inputs
from kaizen.signals import load_signals

log = logging.getLogger(__name__)

# A shift whose busiest hour is forecast below this many transactions is treated as closed.
OPEN_THRESHOLD = 1.0


# ---------------------------------------------------------------- the two rules


def shift_staff(busiest_hour: float, transactions_per_staff_hour: float, min_staff: int, max_staff: int,
                open_threshold: float = OPEN_THRESHOLD) -> tuple[int, int]:
    """(people to roster, people the busiest hour would need without the cap). 0 when closed."""
    if busiest_hour < open_threshold:
        return 0, 0
    need = max(min_staff, int(np.ceil(busiest_hour / transactions_per_staff_hour - 1e-9)))
    return min(need, max_staff), need


def order_quantity(forecast_demand: float, buffer: float, on_hand: float, pack_size: float) -> float:
    shortfall = forecast_demand * (1 + buffer) - on_hand
    if shortfall <= 0:
        return 0.0
    return float(np.ceil(shortfall / pack_size - 1e-9) * pack_size)


# ---------------------------------------------------------------- helpers


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def review_window(inputs: Inputs, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = inputs.history_end
    return end - pd.Timedelta(days=days - 1), end


def day_type(dates: pd.Series, signals: pd.DataFrame) -> pd.Series:
    """'weekend' for Saturdays, Sundays and public holidays, else 'weekday'."""
    ph = set(signals.loc[signals["is_public_holiday"] == 1, "date"])
    weekend = (dates.dt.weekday >= 5) | dates.isin(ph)
    return pd.Series(np.where(weekend, "weekend", "weekday"), index=dates.index)


def hourly_frame(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.assign(date=df["timestamp"].dt.normalize(), hour=df["timestamp"].dt.hour)[["date", "outlet", "hour", value]]


def by_shift(client: ClientConfig, hourly: pd.DataFrame, value: str, roster: pd.DataFrame | None) -> pd.DataFrame:
    """Hourly transactions (forecast or actual) -> one row per date, outlet and shift, with staff."""
    rows = []
    for sh in client.shifts:
        part = hourly[(hourly["hour"] >= sh.start) & (hourly["hour"] < sh.end)]
        g = part.groupby(["date", "outlet"], as_index=False).agg(total=(value, "sum"), busiest=(value, "max"))
        g["shift"], g["start"], g["end"], g["shift_hours"] = sh.name, sh.start, sh.end, sh.hours
        if roster is not None:
            r = roster[(roster["hour"] >= sh.start) & (roster["hour"] < sh.end)]
            on_shift = r.groupby(["date", "outlet"], as_index=False)["scheduled_staff"].max()  # people on the shift
            g = g.merge(on_shift, on=["date", "outlet"], how="outer")
            g[["shift", "start", "end", "shift_hours"]] = [sh.name, sh.start, sh.end, sh.hours]
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df[["total", "busiest"]] = df[["total", "busiest"]].fillna(0)
    df["scheduled_staff"] = df["scheduled_staff"].fillna(0).astype(int) if roster is not None else pd.NA
    staff = [shift_staff(r.busiest, **client.staffing_for(r.outlet).model_dump()) for r in df.itertuples()]
    df["suggested_staff"] = [s for s, _ in staff]
    df["needed_at_peak"] = [n for _, n in staff]
    return df


# ---------------------------------------------------------------- staffing


def staffing_plan(client: ClientConfig, inputs: Inputs, forecast: pd.DataFrame) -> pd.DataFrame:
    roster = inputs.roster
    if roster is not None:
        roster = roster[roster["date"] >= forecast["timestamp"].min().normalize()]
    df = by_shift(client, hourly_frame(forecast, "forecast"), "forecast", roster)
    df = df[df["date"] >= forecast["timestamp"].min().normalize()]
    df = df.rename(columns={"total": "forecast_orders", "busiest": "busiest_hour_orders"})
    df[["forecast_orders", "busiest_hour_orders"]] = df[["forecast_orders", "busiest_hour_orders"]].round(0)
    cols = ["date", "outlet", "shift", "start", "end", "forecast_orders", "busiest_hour_orders",
            "suggested_staff", "needed_at_peak", "scheduled_staff"]
    return df[cols].sort_values(["date", "outlet", "start"]).reset_index(drop=True)


def staffing_gaps(client: ClientConfig, inputs: Inputs) -> pd.DataFrame:
    """Per outlet, day and shift over the review window: rostered vs needed, with hindsight."""
    cols = ["date", "outlet", "shift", "rostered_staff", "needed_staff", "rostered_staff_hours", "needed_staff_hours",
            "overstaffed_hours", "understaffed_hours"]
    if inputs.roster is None:
        return pd.DataFrame(columns=cols)
    start, end = review_window(inputs, client.review_days)
    sales = hourly_frame(inputs.sales, "transactions")
    sales = sales[(sales["date"] >= start) & (sales["date"] <= end)]
    roster = inputs.roster[(inputs.roster["date"] >= start) & (inputs.roster["date"] <= end)]
    df = by_shift(client, sales, "transactions", roster)
    df["rostered_staff"], df["needed_staff"] = df["scheduled_staff"], df["suggested_staff"]
    df["rostered_staff_hours"] = df["rostered_staff"] * df["shift_hours"]
    df["needed_staff_hours"] = df["needed_staff"] * df["shift_hours"]
    df["overstaffed_hours"] = (df["rostered_staff_hours"] - df["needed_staff_hours"]).clip(lower=0)
    df["understaffed_hours"] = (df["needed_staff_hours"] - df["rostered_staff_hours"]).clip(lower=0)
    out = df[cols].astype({c: int for c in cols[3:]})
    return out.sort_values(["date", "outlet", "shift"]).reset_index(drop=True)


# ---------------------------------------------------------------- menu items and ingredients


def sku_mix(inputs: Inputs, signals: pd.DataFrame, review_days: int) -> pd.DataFrame:
    """Units of each menu item per transaction, per outlet and day type."""
    cols = ["outlet", "item", "day_type", "units_per_transaction", "source"]
    rows = []
    if inputs.items is not None:
        start, end = review_window(inputs, review_days)
        items = inputs.items[(inputs.items["date"] >= start) & (inputs.items["date"] <= end)].copy()
        sales = inputs.sales.assign(date=inputs.sales["timestamp"].dt.normalize())
        sales = sales[(sales["date"] >= start) & (sales["date"] <= end)]
        txns = sales.groupby(["outlet", "date"], as_index=False)["transactions"].sum()
        txns["day_type"] = day_type(txns["date"], signals)
        items["day_type"] = day_type(items["date"], signals)
        units = items.groupby(["outlet", "item", "day_type"])["units_sold"].sum()
        total = txns.groupby(["outlet", "day_type"])["transactions"].sum()
        for (outlet, item, dtype), u in units.items():
            rows.append({"outlet": outlet, "item": item, "day_type": dtype,
                         "units_per_transaction": u / total[(outlet, dtype)], "source": f"items.csv, last {review_days} days"})
        mix = pd.DataFrame(rows, columns=cols)
        # A day type with no history in the window (e.g. no holidays) borrows the other one.
        for dtype, other in (("weekday", "weekend"), ("weekend", "weekday")):
            have = set(zip(mix.loc[mix["day_type"] == dtype, "outlet"], mix.loc[mix["day_type"] == dtype, "item"]))
            missing = np.array([k not in have for k in zip(mix["outlet"], mix["item"])], dtype=bool)
            borrow = mix[(mix["day_type"] == other).to_numpy() & missing]
            rows += borrow.assign(day_type=dtype).to_dict("records")
    if inputs.menu is not None:
        have = {(r["outlet"], r["item"]) for r in rows}
        for outlet in inputs.outlets:
            for m in inputs.menu.itertuples():
                if (outlet, m.item) not in have:
                    rows += [{"outlet": outlet, "item": m.item, "day_type": d, "units_per_transaction": m.usage_per_transaction,
                              "source": "menu.csv"} for d in ("weekday", "weekend")]
    return pd.DataFrame(rows, columns=cols).sort_values(["outlet", "item", "day_type"]).reset_index(drop=True)


def sku_forecast(forecast: pd.DataFrame, mix: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    daily = forecast.assign(date=forecast["timestamp"].dt.normalize()).groupby(["date", "outlet"], as_index=False)["forecast"].sum()
    daily["day_type"] = day_type(daily["date"], signals)
    out = daily.merge(mix, on=["outlet", "day_type"])
    out["forecast_units"] = (out["forecast"] * out["units_per_transaction"]).round(1)
    return out[["date", "outlet", "item", "forecast_units"]].sort_values(["date", "outlet", "item"]).reset_index(drop=True)


def ingredient_forecast(skus: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    cols = ["date", "outlet", "ingredient", "unit", "forecast_qty"]
    if skus.empty:
        return pd.DataFrame(columns=cols)
    if inputs.recipes is None:
        units = inputs.menu.set_index("item")["unit"].to_dict() if inputs.menu is not None else {}
        out = skus.assign(ingredient=skus["item"], unit=skus["item"].map(units).fillna("units"), forecast_qty=skus["forecast_units"])
        return out[cols]
    missing = sorted(set(skus["item"]) - set(inputs.recipes["item"]))
    if missing:
        log.warning("No recipe for %s; counting each as its own stock item", missing)
    exploded = skus.merge(inputs.recipes, on="item", how="left")
    no_recipe = exploded["ingredient"].isna()
    exploded.loc[no_recipe, ["ingredient", "unit", "qty_per_item"]] = exploded.loc[no_recipe, ["item"]].assign(unit="units", q=1.0).to_numpy()
    exploded["forecast_qty"] = exploded["forecast_units"] * exploded["qty_per_item"].astype(float)
    out = exploded.groupby(["date", "outlet", "ingredient", "unit"], as_index=False)["forecast_qty"].sum()
    out["forecast_qty"] = out["forecast_qty"].round(2)
    return out[cols].sort_values(["date", "outlet", "ingredient"]).reset_index(drop=True)


def suggested_orders(client: ClientConfig, inputs: Inputs, ingredients: pd.DataFrame) -> pd.DataFrame:
    cols = ["item_key", "outlet", "item", "unit", "forecast_demand", "on_hand", "suggested_order"]
    if ingredients.empty:
        log.warning("No items.csv or menu.csv: no suggested orders")
        return pd.DataFrame(columns=cols)
    first_day = ingredients["date"].min()
    latest_stock = {}
    if inputs.stock is not None:
        last = inputs.stock[inputs.stock["date"] == inputs.stock["date"].max()]
        latest_stock = last.set_index(["outlet", "item"])["on_hand"].to_dict()
    rows = []
    for (outlet, ingredient, unit), g in ingredients.groupby(["outlet", "ingredient", "unit"]):
        p = client.item_params(ingredient)
        demand = g.loc[g["date"] < first_day + pd.Timedelta(days=p.cover_days), "forecast_qty"].sum()
        on_hand = 0.0 if p.perishable else latest_stock.get((outlet, ingredient), 0.0)
        rows.append({
            "item_key": f"{slug(outlet)}.{slug(ingredient)}", "outlet": outlet, "item": ingredient, "unit": unit,
            "forecast_demand": round(demand, 2), "on_hand": round(on_hand, 2),
            "suggested_order": order_quantity(demand, p.buffer, on_hand, p.pack_size),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(["outlet", "item"]).reset_index(drop=True)


# ---------------------------------------------------------------- looking back


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


# ---------------------------------------------------------------- run


def run(client: ClientConfig) -> None:
    inputs = load_inputs(client.inputs_dir)
    signals = load_signals(client.outputs_dir)
    out = client.outputs_dir
    path = out / "forecast.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make forecast` first.")
    forecast = pd.read_csv(path, parse_dates=["timestamp"])

    staffing = staffing_plan(client, inputs, forecast)
    mix = sku_mix(inputs, signals, client.review_days)
    skus = sku_forecast(forecast, mix, signals)
    ingredients = ingredient_forecast(skus, inputs)
    orders = suggested_orders(client, inputs, ingredients)
    gaps = staffing_gaps(client, inputs)
    summary = outlet_summary(client, inputs, gaps)
    savings = savings_estimate(client, summary)

    dated = {"staffing_plan": staffing, "sku_forecast": skus, "ingredient_forecast": ingredients, "staffing_gaps": gaps}
    for name, df in dated.items():
        df.to_csv(out / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    mix.round({"units_per_transaction": 4}).to_csv(out / "sku_mix.csv", index=False)
    orders.to_csv(out / "suggested_orders.csv", index=False)
    summary.to_csv(out / "outlet_summary.csv", index=False)
    savings.to_csv(out / "savings_estimate.csv", index=False)

    open_shifts = staffing[staffing["suggested_staff"] > 0]
    print(f"Plan for {client.name}, week of {forecast['timestamp'].min():%d %b %Y} -> {out}")
    print(f"  shifts: {len(open_shifts)} to staff, {open_shifts['suggested_staff'].sum():,} people-shifts"
          + (f" (roster has {staffing['scheduled_staff'].sum():,})" if inputs.roster is not None else "")
          + f", {int((open_shifts['needed_at_peak'] > open_shifts['suggested_staff']).sum())} at the maximum")
    print(f"  menu items forecast: {skus['item'].nunique()}, ingredients: {ingredients['ingredient'].nunique()}, "
          f"order lines: {int((orders['suggested_order'] > 0).sum())}")
    print(f"\nLast {client.review_days} days by outlet:\n{summary.to_string(index=False)}")
    total = savings.iloc[-1]["estimated_saving_sgd_per_month"]
    if pd.notna(total):
        print(f"\nEstimated saving (an estimate from client.yaml assumptions, not a guarantee): about S${total:,.0f} per month")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Turn the forecast into shifts, menu items, ingredients and orders.")
    parser.add_argument("--client", default="demo")
    run(load_client(parser.parse_args().client))


if __name__ == "__main__":
    main()
