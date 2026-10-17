"""Kaizen Labs dashboard: next week for a multi-outlet operator, on one page.

    make dashboard                 # the Demo Café Group sample
    make dashboard CLIENT=<key>    # any client that has been through forecast + plan

Answers four questions, in order: how busy will we be, who do I roster, what will sell,
what do I order. How the forecast was checked sits in collapsed sections at the bottom.
Reads only files in the client's outputs folder; each section names its source file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kaizen.config import load_client  # noqa: E402

ACCENT = "#1D4ED8"
MUTED = "#8d8c86"
STAFF_COLORS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]  # 3, 4, 5, 6 people: one blue, light to dark
ALL = "All outlets"
WHOLE_UNITS = {"pcs", "slices", "units"}

st.set_page_config(page_title="Kaizen Labs | Next week", page_icon="☕", layout="centered")


def client_key() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="demo")
    return parser.parse_known_args(sys.argv[1:])[0].client


@st.cache_data
def load(outputs_dir: str, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(outputs_dir) / name, **kwargs)


def source(*files: str) -> None:
    st.caption("Source: " + ", ".join(f"`{f}`" for f in files))


def short(outlet: str) -> str:
    return outlet.split(" - ")[0]


def fmt_qty(value: float, unit: str, packs: bool = False) -> str:
    """Whole numbers for counted units (and for order quantities that are whole packs), else one decimal."""
    whole = unit in WHOLE_UNITS or (packs and float(value).is_integer())
    return f"{value:,.0f}" if whole else f"{value:,.1f}"


def fit(n_rows: int) -> int:
    """Table height that shows every row without an inner scrollbar."""
    return 35 * (n_rows + 1) + 3


client = load_client(client_key())
out = str(client.outputs_dir)
try:
    daily = load(out, "daily_sales.csv", parse_dates=["date"], keep_default_na=False, na_values=[""])
    shifts = load(out, "staffing_plan.csv", parse_dates=["date"])
    skus = load(out, "sku_forecast.csv", parse_dates=["date"])
    ingredients = load(out, "ingredient_forecast.csv", parse_dates=["date"])
    orders = load(out, "suggested_orders.csv")
    acc = load(out, "backtest_accuracy.csv")
    summary = load(out, "outlet_summary.csv")
    savings = load(out, "savings_estimate.csv")
except FileNotFoundError as exc:
    st.error(f"Missing output file: {exc.filename}. Run `make forecast plan CLIENT={client.key}` first.")
    st.stop()

week_start = shifts["date"].min()
days = pd.date_range(week_start, periods=7, freq="D")
day_labels = [f"{d:%a %d}" for d in days]
has_roster = shifts["scheduled_staff"].notna().any()
shift_label = {x.start: x.label for x in client.shifts}

# ---------------------------------------------------------------- header
st.markdown(
    "<div style='font-family:\"JetBrains Mono\",ui-monospace,monospace;font-weight:700;font-size:1.1rem'>"
    f"kaizen<span style='color:{ACCENT}'>.</span>labs</div>",
    unsafe_allow_html=True,
)
st.title(f"Next week: {week_start:%d} to {days[-1]:%d %b %Y}")
st.markdown(f"**{client.name}**" + (" · sample data for a fictional business" if client.is_sample else ""))
outlet = st.selectbox("Outlet", [ALL, *sorted(daily["outlet"].unique())], label_visibility="collapsed")


def pick(df: pd.DataFrame) -> pd.DataFrame:
    return df if outlet == ALL else df[df["outlet"] == outlet]


# ---------------------------------------------------------------- headline numbers
week = pick(daily[daily["date"] >= week_start]).groupby("date")[["forecast", "last_week_rule"]].sum()
s = pick(shifts)
open_shifts = s[s["suggested_staff"] > 0]
o = pick(orders)
change = week["forecast"].sum() / week["last_week_rule"].sum() - 1

def tile(col, label: str, value: str, note: str) -> None:
    col.metric(label, value)
    col.caption(note)


c1, c2 = st.columns(2)
busiest = week["forecast"].idxmax()
tile(c1, "Customers expected", f"{week['forecast'].sum():,.0f}", f"{change:+.1%} vs last week")
tile(c2, "Busiest day", f"{busiest:%A}", f"{week.loc[busiest, 'forecast']:,.0f} customers")
c3, c4 = st.columns(2)
people = int(open_shifts["suggested_staff"].sum())
tile(c3, "People-shifts to roster", f"{people:,}",
     f"current roster has {int(s['scheduled_staff'].sum()):,}" if has_roster else "no roster loaded")
tile(c4, "Ingredients to order", f"{int((o['suggested_order'] > 0).sum())} lines", "see the order list below")
st.caption("Customers = transactions at the till.")

# Things that need a decision this week.
notes = []
holidays = daily[(daily["date"] >= week_start) & (daily["public_holiday"].fillna("") != "")].drop_duplicates("date")
for _, r in holidays.iterrows():
    notes.append(f"**{r['date']:%a %d %b} is {r['public_holiday']}.** Already in the forecast and the shifts.")
capped = open_shifts[open_shifts["needed_at_peak"] > open_shifts["suggested_staff"]]
for (o_name, sh), g in capped.groupby(["outlet", "shift"]):
    days_txt = ", ".join(f"{d:%a}" for d in g["date"])
    notes.append(f"**{o_name}, {sh} shift ({days_txt}):** the busiest hour needs up to {g['needed_at_peak'].max()} people "
                 f"but shifts stop at {client.staffing_for(o_name).max_staff}. Consider a part-timer for the peak hour.")
if notes:
    st.markdown("#### Needs a decision")
    st.markdown("\n".join(f"- {n}" for n in notes))

# ---------------------------------------------------------------- 1. how busy
st.subheader("1. How busy each day will be")
w = week.reset_index()
w["day"] = w["date"].dt.strftime("%a %d")
base = alt.Chart(w).encode(x=alt.X("day:N", sort=day_labels, title=None, axis=alt.Axis(labelAngle=0)))
bars = base.mark_bar(color=ACCENT, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=28).encode(
    y=alt.Y("forecast:Q", title="Customers", axis=alt.Axis(format=",.0f")),
    tooltip=[alt.Tooltip("day:N", title="day"), alt.Tooltip("forecast:Q", title="forecast", format=",.0f"),
             alt.Tooltip("last_week_rule:Q", title="same day last week", format=",.0f")],
)
labels = base.mark_text(dy=-8, fontSize=11, color=MUTED).encode(y="forecast:Q", text=alt.Text("forecast:Q", format=",.0f"))
last_week = base.mark_tick(color="#111418", thickness=2, size=34, opacity=0.55).encode(y="last_week_rule:Q")
st.altair_chart((bars + last_week + labels).properties(height=240), width="stretch")
st.caption("Bars: forecast customers. Dark tick: the same day last week.")
source("daily_sales.csv")

# ---------------------------------------------------------------- 2. shifts
st.subheader("2. Who to roster")
grid = s[s["suggested_staff"] > 0].copy()
grid["day"] = grid["date"].dt.strftime("%a %d")
grid["shift_label"] = grid["start"].map(shift_label)
grid["row"] = grid["shift_label"] if outlet != ALL else grid["outlet"].map(short) + " · " + grid["shift"]
row_order = ([x.label for x in client.shifts] if outlet != ALL else
             [f"{short(o_)} · {x.name}" for o_ in sorted(grid["outlet"].unique()) for x in client.shifts])
grid["text"] = grid["suggested_staff"].astype(str) + (grid["needed_at_peak"] > grid["suggested_staff"]).map({True: "+", False: ""})
lo, hi = client.staffing.min_staff, client.staffing.max_staff
cell = alt.Chart(grid).encode(
    x=alt.X("day:N", sort=day_labels, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
    y=alt.Y("row:N", sort=[r for r in row_order if r in set(grid["row"])], title=None, axis=alt.Axis(labelLimit=220)),
    tooltip=[alt.Tooltip("date:T", title="day", format="%a %d %b"), "outlet:N", alt.Tooltip("shift_label:N", title="shift"),
             alt.Tooltip("suggested_staff:Q", title="people"), alt.Tooltip("scheduled_staff:Q", title="on the roster now"),
             alt.Tooltip("busiest_hour_orders:Q", title="busiest hour, customers"),
             alt.Tooltip("forecast_orders:Q", title="shift total, customers", format=",.0f")],
)
heat = cell.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
    color=alt.Color("suggested_staff:O", title="People on shift", legend=alt.Legend(orient="bottom"),
                    scale=alt.Scale(domain=list(range(lo, hi + 1)), range=(STAFF_COLORS * 2)[:hi - lo + 1])))
text = cell.mark_text(fontSize=12, fontWeight="bold").encode(
    text="text:N", color=alt.condition(alt.datum.suggested_staff >= lo + 2, alt.value("white"), alt.value("#111418")))
st.altair_chart((heat + text).properties(height=30 * grid["row"].nunique() + 40), width="stretch")
st.caption(
    f"People needed per shift, {lo} to {hi}: enough for the shift's busiest hour at "
    f"{client.staffing.transactions_per_staff_hour:.0f} customers per person per hour. "
    "**+** = the busiest hour needs more than the maximum. Blank = closed."
)
if has_roster:
    diff = s[s["suggested_staff"] != s["scheduled_staff"]].copy()
    if diff.empty:
        st.markdown("The current roster already matches.")
    else:
        diff["Change"] = (diff["suggested_staff"] - diff["scheduled_staff"]).map("{:+d}".format)
        diff["Day"] = diff["date"].dt.strftime("%a %d %b")
        diff["Shift"] = diff["start"].map(shift_label)
        table = diff.rename(columns={"outlet": "Outlet", "scheduled_staff": "Rostered", "suggested_staff": "Suggested"})
        cols = ["Day", "Outlet", "Shift", "Rostered", "Suggested", "Change"]
        if outlet != ALL:
            cols.remove("Outlet")
        add = int((diff["suggested_staff"] > diff["scheduled_staff"]).sum())
        st.markdown(f"**Changes to the current roster: {len(diff)} shifts.** {add} need more people, "
                    f"{len(diff) - add} can run with fewer (0 = no customers expected on that shift).")
        st.dataframe(table[cols], hide_index=True, width="stretch", height=min(35 * len(diff) + 38, 320))
source("staffing_plan.csv")

# ---------------------------------------------------------------- 3. what will sell
st.subheader("3. What will sell")
k = pick(skus).assign(day=lambda d: d["date"].dt.strftime("%a %d"))
k = k.pivot_table(index="item", columns="day", values="forecast_units", aggfunc="sum")[day_labels]
k["Week"] = k.sum(axis=1)
k = k.sort_values("Week", ascending=False).round(0).astype(int)
k.index.name = "Menu item"
st.dataframe(k, width="stretch", height=fit(len(k)))
st.caption("Forecast customers × each item's share of sales at that outlet over the last 4 weeks "
           "(weekdays and weekends/holidays counted separately).")
source("sku_forecast.csv")

# ---------------------------------------------------------------- 4. ingredients and orders
st.subheader("4. Ingredients and what to order")
g = pick(ingredients).assign(day=lambda d: d["date"].dt.strftime("%a %d"))
use = g.pivot_table(index=["ingredient", "unit"], columns="day", values="forecast_qty", aggfunc="sum")[day_labels]
use["Week"] = use.sum(axis=1)
use = use.reset_index()
shown = use.copy()
for col in [*day_labels, "Week"]:
    shown[col] = [fmt_qty(v, u) for v, u in zip(use[col], use["unit"])]
st.markdown("**Daily use, forecast**")
st.dataframe(shown.rename(columns={"ingredient": "Ingredient", "unit": "Unit"}), hide_index=True, width="stretch",
             height=fit(len(shown)))
st.caption("Menu items × recipes. Includes everything a recipe uses, e.g. a flat white = 18 g beans, 150 ml milk, a cup and lid if takeaway.")
source("ingredient_forecast.csv")

st.markdown("**Order list**")
to_order = o[o["suggested_order"] > 0]
if outlet == ALL:
    pv = to_order.assign(qty=[fmt_qty(v, u, packs=True) for v, u in zip(to_order["suggested_order"], to_order["unit"])])
    pv = pv.pivot_table(index=["item", "unit"], columns="outlet", values="qty", aggfunc="first").fillna("–").reset_index()
    pv.columns = ["Ingredient", "Unit", *[short(c) for c in pv.columns[2:]]]
    st.dataframe(pv, hide_index=True, width="stretch", height=fit(len(pv)))
else:
    t = to_order.copy()
    for col in ("forecast_demand", "on_hand", "suggested_order"):
        t[col] = [fmt_qty(v, u, packs=col == "suggested_order") for v, u in zip(t[col], t["unit"])]
    t["Covers"] = t["item"].map(lambda i: f"{client.item_params(i).cover_days} day" + ("s" if client.item_params(i).cover_days > 1 else ""))
    t = t.rename(columns={"item": "Ingredient", "unit": "Unit", "forecast_demand": "Will use", "on_hand": "In stock",
                          "suggested_order": "Order"})
    st.dataframe(t[["Ingredient", "Unit", "Covers", "Will use", "In stock", "Order"]], hide_index=True, width="stretch",
                 height=fit(len(t)))
st.caption("Order = forecast use until the next delivery, plus a safety buffer, minus what's in stock, rounded up to "
           "whole packs. Fresh items (bread, lettuce, banana bread) are ordered for tomorrow only.")
source("suggested_orders.csv")

# ---------------------------------------------------------------- the back-office detail
st.divider()
with st.expander("How reliable is the forecast?"):
    a = acc[(acc["outlet"] == outlet) & (acc["level"] == "day")].set_index("method")["accuracy"]
    st.markdown(
        f"Over the last 8 weeks, forecasts made a week ahead were **{a['model']:.0%} accurate** on daily customers, "
        f"against {a['last_week_rule']:.0%} for \"same as last week\". Accuracy = 1 − (total error ÷ total customers)."
    )
    t = acc[acc["level"] == "day"].pivot(index="outlet", columns="method", values="accuracy")
    t = t.loc[[*sorted(i for i in t.index if i != ALL), ALL], ["model", "last_week_rule"]]
    t.columns = ["Forecast", "Same as last week"]
    st.dataframe(t.style.format("{:.0%}"), width="stretch")
    source("backtest_accuracy.csv")

with st.expander(f"Last {client.review_days} days: waste, roster gaps, stock-outs"):
    st.dataframe(
        summary.rename(columns={"outlet": "Outlet", "waste_sgd": "Waste (S$)", "overstaffed_hours": "Over-staffed (staff-hours)",
                                "understaffed_hours": "Under-staffed (staff-hours)", "stockouts": "Ran out (item-days)"}),
        hide_index=True, width="stretch", column_config={"Waste (S$)": st.column_config.NumberColumn(format="%.0f")})
    row = savings[savings["outlet"] == outlet].iloc[0]
    st.markdown(
        f"**Savings estimate: about S${row['estimated_saving_sgd_per_month']:,.0f} a month** (an estimate, not a guarantee): "
        f"S${row['avoidable_waste_sgd_per_month']:,.0f} if {row['avoidable_waste_share']:.0%} of waste is avoided, plus "
        f"S${row['recoverable_wages_sgd_per_month']:,.0f} if {row['recoverable_overstaff_share']:.0%} of over-staffed hours are "
        f"cut at S${row['hourly_wage_sgd']:.2f}/hour. The assumptions are agreed with you."
    )
    source("outlet_summary.csv", "savings_estimate.csv")
