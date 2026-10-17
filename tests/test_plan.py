import numpy as np
import pandas as pd
import pytest

from kaizen.contract import Inputs
from kaizen.plan import (
    ingredient_forecast, order_quantity, outlet_summary, shift_staff, sku_forecast, sku_mix, staffing_gaps,
    staffing_plan, suggested_orders,
)
from tests.conftest import calendar, hourly_sales


# ---- shifts

@pytest.mark.parametrize("busiest, expected", [
    (0.0, (0, 0)),       # no trade expected: shift closed
    (0.5, (0, 0)),       # below the open threshold
    (20, (3, 3)),        # quiet: never below the minimum of 3
    (91, (4, 4)),        # ceil(91 / 30) = 4
    (150, (5, 5)),       # exactly 5 x 30: no extra person
    (151, (6, 6)),
    (200, (6, 7)),       # needs 7, capped at 6, and the 7 is reported
])
def test_shift_staff(busiest, expected):
    assert shift_staff(busiest, transactions_per_staff_hour=30, min_staff=3, max_staff=6) == expected


def hours_forecast(outlet: str, day: str, per_hour: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in per_hour],
                         "outlet": outlet, "forecast": list(per_hour.values())})


def test_staffing_plan_sizes_each_shift_for_its_busiest_hour(client):
    fc = hours_forecast("A", "2026-01-08", {8: 40, 9: 130, 10: 60, 12: 80, 13: 200, 19: 10, 20: 5})
    roster = pd.DataFrame({"date": pd.Timestamp("2026-01-08"), "outlet": "A", "hour": [8, 9, 12, 13, 19],
                           "scheduled_staff": [4, 5, 4, 4, 4]})
    plan = staffing_plan(client, Inputs(sales=hourly_sales(), roster=roster), fc).set_index("shift")
    assert plan.loc["Opening", "suggested_staff"] == 5          # ceil(130 / 30)
    assert plan.loc["Opening", "forecast_orders"] == 230
    assert plan.loc["Opening", "scheduled_staff"] == 5          # people on the shift = most rostered in any hour
    assert plan.loc["Mid", "suggested_staff"] == 6               # 200 needs 7, capped at 6
    assert plan.loc["Mid", "needed_at_peak"] == 7
    assert plan.loc["Closing", "suggested_staff"] == 3           # quiet evening: the minimum
    assert plan.loc["Closing", "scheduled_staff"] == 4


def test_staffing_plan_uses_per_outlet_settings(client):
    client.staffing_by_outlet = {"B": {"min_staff": 4, "transactions_per_staff_hour": 20}}
    fc = pd.concat([hours_forecast("A", "2026-01-08", {9: 50}), hours_forecast("B", "2026-01-08", {9: 50})])
    plan = staffing_plan(client, Inputs(sales=hourly_sales()), fc).set_index("outlet")
    assert plan.loc["A", "suggested_staff"] == 3   # ceil(50 / 30) = 2 -> minimum 3
    assert plan.loc["B", "suggested_staff"] == 4   # ceil(50 / 20) = 3 -> B's minimum 4
    assert plan["scheduled_staff"].isna().all()    # no roster provided


def test_rostered_shift_with_no_trade_shows_as_closed(client):
    fc = hours_forecast("A", "2026-01-08", {9: 50})
    roster = pd.DataFrame({"date": pd.Timestamp("2026-01-08"), "outlet": "A", "hour": [9, 19], "scheduled_staff": [3, 3]})
    plan = staffing_plan(client, Inputs(sales=hourly_sales(), roster=roster), fc).set_index("shift")
    assert plan.loc["Closing", "suggested_staff"] == 0 and plan.loc["Closing", "scheduled_staff"] == 3


def test_staffing_gaps_count_staff_hours_per_shift(client):
    sales = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-05 09:00", "2026-01-05 13:00"]), "outlet": "A",
                          "transactions": [140, 30]})
    # needed: Opening 5 (ceil 140/30), Mid 3, Closing 0 (no trade). Rostered: 4, 4, 3.
    roster = pd.DataFrame({"date": pd.Timestamp("2026-01-05"), "outlet": "A", "hour": [9, 13, 19], "scheduled_staff": [4, 4, 3]})
    gaps = staffing_gaps(client, Inputs(sales=sales, roster=roster)).set_index("shift")
    assert gaps.loc["Opening", "understaffed_hours"] == 6     # one person short for 6 hours
    assert gaps.loc["Mid", "overstaffed_hours"] == 6          # one extra for 6 hours
    assert gaps.loc["Closing", "overstaffed_hours"] == 18     # three people, nobody to serve
    assert gaps["rostered_staff_hours"].sum() == (4 + 4 + 3) * 6


# ---- menu items -> ingredients -> orders

def items_inputs() -> Inputs:
    # Mon 5 - Sun 11 Jan 2026, 40 transactions a day; flat whites sell 0.5 per transaction on weekdays, 0.25 at weekends.
    sales = hourly_sales(days=7, outlets=("A",), hours=range(8, 12), value=10)
    dates = pd.date_range("2026-01-05", periods=7, freq="D")
    items = pd.DataFrame({"date": dates, "outlet": "A", "item": "Flat White",
                          "units_sold": [20.0 if d.weekday() < 5 else 10.0 for d in dates]})
    recipes = pd.DataFrame({"item": ["Flat White", "Flat White"], "ingredient": ["Beans", "Milk"],
                            "unit": ["kg", "L"], "qty_per_item": [0.02, 0.15]})
    menu = pd.DataFrame({"item": ["Flat White", "Croissant"], "unit": ["pcs", "pcs"], "usage_per_transaction": [0.4, 0.3]})
    stock = pd.DataFrame({"date": pd.Timestamp("2026-01-11"), "outlet": "A", "item": ["Beans", "Milk"],
                          "unit": ["kg", "L"], "on_hand": [1.0, 2.0]})
    return Inputs(sales=sales, items=items, recipes=recipes, menu=menu, stock=stock)


def week_forecast(per_hour: float = 10.0, start: str = "2026-01-12") -> pd.DataFrame:
    ts = [pd.Timestamp(start) + pd.Timedelta(days=d, hours=h) for d in range(7) for h in range(8, 12)]
    return pd.DataFrame({"timestamp": ts, "outlet": "A", "forecast": per_hour})


def test_sku_mix_splits_weekdays_and_weekends_and_falls_back_to_menu():
    mix = sku_mix(items_inputs(), calendar(), 28).set_index(["item", "day_type"])
    assert mix.loc[("Flat White", "weekday"), "units_per_transaction"] == pytest.approx(0.5)
    assert mix.loc[("Flat White", "weekend"), "units_per_transaction"] == pytest.approx(0.25)
    assert mix.loc[("Croissant", "weekday"), "source"] == "menu.csv"   # no sales history: agreed rate


def test_public_holiday_counts_as_a_weekend_day():
    signals = calendar(holidays=["2026-01-13"])
    skus = sku_forecast(week_forecast(), sku_mix(items_inputs(), calendar(), 28), signals)
    fw = skus[skus["item"] == "Flat White"].set_index("date")["forecast_units"]
    assert fw[pd.Timestamp("2026-01-12")] == 20      # Monday: 40 x 0.5
    assert fw[pd.Timestamp("2026-01-13")] == 10      # holiday Tuesday: 40 x 0.25


def test_ingredients_are_menu_items_times_recipes():
    skus = pd.DataFrame({"date": pd.Timestamp("2026-01-12"), "outlet": "A", "item": ["Flat White"], "forecast_units": [100.0]})
    ing = ingredient_forecast(skus, items_inputs()).set_index("ingredient")
    assert ing.loc["Beans", "forecast_qty"] == pytest.approx(2.0)    # 100 x 20 g
    assert ing.loc["Milk", "forecast_qty"] == pytest.approx(15.0)    # 100 x 150 ml
    assert ing.loc["Milk", "unit"] == "L"


def test_item_without_a_recipe_is_its_own_stock_item():
    skus = pd.DataFrame({"date": pd.Timestamp("2026-01-12"), "outlet": "A", "item": ["Croissant"], "forecast_units": [30.0]})
    ing = ingredient_forecast(skus, items_inputs())
    assert ing[["ingredient", "forecast_qty"]].values.tolist() == [["Croissant", 30.0]]


def test_orders_cover_days_buffer_stock_and_packs(client):
    inputs = items_inputs()
    signals = calendar()
    skus = sku_forecast(week_forecast(), sku_mix(inputs, signals, 28), signals)
    orders = suggested_orders(client, inputs, ingredient_forecast(skus, inputs)).set_index("item")
    # Beans, 7 days: flat whites = 5 x 20 + 2 x 10 = 120 -> 2.4 kg; x 1.1 = 2.64; minus 1 kg = 1.64 -> 2 bags
    assert orders.loc["Beans", "forecast_demand"] == pytest.approx(2.4)
    assert orders.loc["Beans", "suggested_order"] == 2
    # Milk, 3 days (Mon-Wed): 60 flat whites x 0.15 = 9 L; minus 2 L = 7 -> one case of 12
    assert orders.loc["Milk", "forecast_demand"] == pytest.approx(9.0)
    assert orders.loc["Milk", "suggested_order"] == 12
    # Croissant: perishable, 1 day, 40 x 0.3 = 12 -> exactly one pack of 12
    assert orders.loc["Croissant", "on_hand"] == 0
    assert orders.loc["Croissant", "suggested_order"] == 12


@pytest.mark.parametrize("demand, buffer, on_hand, pack, expected", [
    (100, 0.10, 0, 1, 110), (100, 0.10, 30, 1, 80), (100, 0.10, 0, 12, 120), (96, 0.0, 0, 12, 96),
    (100, 0.10, 200, 12, 0), (0.3, 0.10, 0, 1, 1), (2.5, 0.0, 0, 0.5, 2.5),
])
def test_order_quantity(demand, buffer, on_hand, pack, expected):
    assert order_quantity(demand, buffer, on_hand, pack) == pytest.approx(expected)


# ---- looking back

def test_outlet_summary_counts_stockouts_and_waste_in_review_window(client):
    client.review_days = 2
    sales = hourly_sales(days=3, outlets=("A",))
    stock = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]), "outlet": "A",
                          "item": "Beans", "unit": "kg", "on_hand": [0.0, 0.0, 3.0]})
    waste = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]), "outlet": "A",
                          "waste_sgd": [100.0, 10.0, 5.5]})
    inputs = Inputs(sales=sales, stock=stock, waste=waste)
    summary = outlet_summary(client, inputs, staffing_gaps(client, inputs)).iloc[0]
    assert summary.stockouts == 1           # 5 Jan is outside the 2-day window
    assert summary.waste_sgd == pytest.approx(15.5)
    assert np.isnan(summary.overstaffed_hours)  # no roster provided
