import numpy as np
import pandas as pd
import pytest

from kaizen.contract import Inputs
from kaizen.plan import (
    item_rates, order_quantity, outlet_summary, staff_needed, staffing_gaps, staffing_plan, suggested_orders,
)
from tests.conftest import hourly_sales


# ---- orders

@pytest.mark.parametrize("demand, buffer, on_hand, pack, expected", [
    (100, 0.10, 0, 1, 110),     # demand plus 10% buffer
    (100, 0.10, 30, 1, 80),     # minus stock on hand
    (100, 0.10, 0, 12, 120),    # 110 rounded up to whole packs of 12
    (96, 0.0, 0, 12, 96),       # exactly 8 packs: no extra pack
    (100, 0.10, 200, 12, 0),    # enough stock: order nothing, never negative
    (0.3, 0.10, 0, 1, 1),       # any shortfall orders at least one pack
    (2.5, 0.0, 0, 0.5, 2.5),    # fractional pack sizes (e.g. 0.5 kg)
])
def test_order_quantity(demand, buffer, on_hand, pack, expected):
    assert order_quantity(demand, buffer, on_hand, pack) == pytest.approx(expected)


def menu_inputs(stock_on_hand: float) -> Inputs:
    sales = hourly_sales(days=3, outlets=("A",), hours=range(8, 12), value=10)  # 40 txns/day
    menu = pd.DataFrame({"item": ["Beans", "Croissant"], "unit": ["kg", "pcs"], "usage_per_transaction": [0.02, 0.5]})
    stock = pd.DataFrame({"date": pd.to_datetime(["2026-01-07"] * 2), "outlet": ["A", "A"], "item": ["Beans", "Croissant"],
                          "unit": ["kg", "pcs"], "on_hand": [stock_on_hand, 50.0]})
    return Inputs(sales=sales, menu=menu, stock=stock)


def week_forecast(per_hour: float = 10.0) -> pd.DataFrame:
    ts = [pd.Timestamp("2026-01-08") + pd.Timedelta(days=d, hours=h) for d in range(7) for h in range(8, 12)]
    return pd.DataFrame({"timestamp": ts, "outlet": "A", "forecast": per_hour})


def test_suggested_orders_use_cover_days_buffer_and_stock(client):
    inputs = menu_inputs(stock_on_hand=1.0)
    orders = suggested_orders(client, inputs, week_forecast(), item_rates(inputs, 28)).set_index("item")
    # Beans: 7 days x 40 txns x 0.02 kg = 5.6 kg; x 1.1 = 6.16; minus 1 kg on hand = 5.16 -> 6 bags of 1 kg
    assert orders.loc["Beans", "forecast_demand"] == pytest.approx(5.6)
    assert orders.loc["Beans", "suggested_order"] == 6
    # Croissant: perishable, 1 day x 40 x 0.5 = 20; the 50 left over are written off, not counted -> 2 packs of 12
    assert orders.loc["Croissant", "on_hand"] == 0
    assert orders.loc["Croissant", "suggested_order"] == 24


def test_items_csv_rates_override_menu(client):
    inputs = menu_inputs(stock_on_hand=0.0)
    inputs.items = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]), "outlet": "A",
                                 "item": "Beans", "units_sold": [1.2, 1.2, 1.2]})  # 3.6 kg over 120 txns
    rates = item_rates(inputs, 28).set_index("item")
    assert rates.loc["Beans", "usage_per_transaction"] == pytest.approx(0.03)
    assert rates.loc["Beans", "source"].startswith("items.csv")
    assert rates.loc["Croissant", "source"] == "menu.csv"


# ---- staffing

def test_staff_needed_rounds_up_with_minimum_and_closed_hours():
    t = [0, 0.4, 1, 20, 21, 59, 61]
    assert staff_needed(t, transactions_per_staff_hour=20, min_staff=2).tolist() == [0, 0, 2, 2, 2, 3, 4]


def test_staff_needed_exact_multiple_does_not_add_a_person():
    assert staff_needed([40.0, 40.000001], 20, 1).tolist() == [2, 3]


def test_staffing_plan_uses_per_outlet_settings_and_roster(client):
    client.staffing_by_outlet = {"A": {"transactions_per_staff_hour": 10, "min_staff": 1}}
    fc = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-08 08:00", "2026-01-08 09:00"] * 2),
                       "outlet": ["A", "A", "B", "B"], "forecast": [25.0, 5.0, 25.0, 5.0]})
    roster = pd.DataFrame({"date": pd.to_datetime(["2026-01-08"]), "outlet": ["A"], "hour": [8], "scheduled_staff": [4]})
    plan = staffing_plan(client, Inputs(sales=hourly_sales(), roster=roster), fc).set_index(["outlet", "hour"])
    assert plan.loc[("A", 8), "suggested_staff"] == 3   # ceil(25 / 10)
    assert plan.loc[("A", 9), "suggested_staff"] == 1   # min 1 for A
    assert plan.loc[("B", 8), "suggested_staff"] == 2   # ceil(25 / 20) = 2, default min 2
    assert plan.loc[("A", 8), "scheduled_staff"] == 4
    assert plan.loc[("B", 9), "scheduled_staff"] == 0   # nobody rostered


def test_staffing_gaps_count_over_and_under_staffed_hours(client):
    sales = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 09:00", "2026-01-05 10:00"]),
                          "outlet": "A", "transactions": [80, 20, 0]})
    # needed: 4, 2, 0 (closed)   rostered: 2, 3, 2
    roster = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"] * 3), "outlet": "A", "hour": [8, 9, 10],
                           "scheduled_staff": [2, 3, 2]})
    gaps = staffing_gaps(client, Inputs(sales=sales, roster=roster))
    row = gaps.iloc[0]
    assert (row.rostered_staff_hours, row.needed_staff_hours) == (7, 6)
    assert row.overstaffed_hours == 1 + 2   # 09:00 one extra, 10:00 two people with no customers
    assert row.understaffed_hours == 2      # 08:00 two short


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
