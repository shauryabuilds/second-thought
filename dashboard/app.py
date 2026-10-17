"""Kaizen Labs demo dashboard.

    make dashboard                 # the Demo Café Group sample
    make dashboard CLIENT=<key>    # any client that has been through forecast + plan

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
INK = "#52514e"  # actuals: neutral ink so the forecast stays the colored line
MUTED = "#a3a29c"
ALL = "All outlets"

st.set_page_config(page_title="Kaizen Labs | Demo", page_icon="☕", layout="centered")


def client_key() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="demo")
    return parser.parse_known_args(sys.argv[1:])[0].client


@st.cache_data
def load(outputs_dir: str, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(outputs_dir) / name, **kwargs)


def source(*files: str) -> None:
    st.caption("Source: " + ", ".join(f"`data/.../{f}`" for f in files))


client = load_client(client_key())
out = str(client.outputs_dir)
try:
    daily = load(out, "daily_sales.csv", parse_dates=["date"], keep_default_na=False, na_values=[""])
    acc = load(out, "backtest_accuracy.csv")
    staffing = load(out, "staffing_plan.csv", parse_dates=["date"])
    orders = load(out, "suggested_orders.csv")
    summary = load(out, "outlet_summary.csv")
    savings = load(out, "savings_estimate.csv")
except FileNotFoundError as exc:
    st.error(f"Missing output file: {exc.filename}. Run `make forecast plan CLIENT={client.key}` first.")
    st.stop()

week_start = staffing["date"].min()

# ---------------------------------------------------------------- header
st.markdown(
    "<div style='font-family:\"JetBrains Mono\",ui-monospace,monospace;font-weight:700;font-size:1.1rem'>"
    f"kaizen<span style='color:{ACCENT}'>.</span>labs</div>",
    unsafe_allow_html=True,
)
st.title(client.name)
badge = " · **sample data, fictional business**" if client.is_sample else ""
st.markdown(f"Plan for the week of **{week_start:%a %d %b %Y}**{badge}")

outlets = sorted(daily["outlet"].unique())
outlet = st.selectbox("Outlet", [ALL, *outlets], label_visibility="collapsed")


def pick(df: pd.DataFrame) -> pd.DataFrame:
    return df if outlet == ALL else df[df["outlet"] == outlet]


# ---------------------------------------------------------------- headline accuracy
def accuracy(level: str, method: str) -> float:
    row = acc[(acc["level"] == level) & (acc["outlet"] == outlet) & (acc["method"] == method)]
    return float(row["accuracy"].iloc[0])


weeks = (daily["actual"].notna().sum() // max(len(outlets), 1)) // 7
model_h, rule_h = accuracy("hour", "model"), accuracy("hour", "last_week_rule")
model_d, rule_d = accuracy("day", "model"), accuracy("day", "last_week_rule")
c1, c2 = st.columns(2)
c1.metric("Forecast accuracy, hour by hour", f"{model_h:.0%}", f"{(model_h - rule_h) * 100:+.1f} pts vs last-week rule")
c2.metric("Forecast accuracy, daily totals", f"{model_d:.0%}", f"{(model_d - rule_d) * 100:+.1f} pts vs last-week rule")
st.caption(
    f"Back-tested over the last {weeks} weeks, always forecasting 7 days ahead without seeing the answer. "
    "Accuracy = 1 − (total error ÷ total transactions). The \"last-week rule\" assumes each hour repeats "
    "the same hour last week."
)
with st.expander("Accuracy by outlet"):
    table = acc[acc["level"] == "hour"].pivot(index="outlet", columns="method", values="accuracy")
    table = table.loc[[*outlets, ALL], ["model", "last_week_rule", "four_week_avg"]]
    table.columns = ["Forecast", "Same as last week", "4-week average"]
    st.dataframe(table.style.format("{:.1%}"), use_container_width=True)
    source("backtest_accuracy.csv")

# ---------------------------------------------------------------- daily actual vs forecast
st.subheader("Daily transactions")
d = pick(daily).groupby("date", as_index=False)[["actual", "forecast", "last_week_rule"]].sum(min_count=1)
d = d.merge(daily.drop_duplicates("date")[["date", "rain_mm", "public_holiday"]], on="date")
long = d.melt(id_vars=["date", "rain_mm", "public_holiday"], value_vars=["actual", "forecast", "last_week_rule"],
              var_name="series", value_name="transactions").dropna(subset=["transactions"])
long["series"] = long["series"].map({"actual": "Actual", "forecast": "Forecast", "last_week_rule": "Same as last week"})
color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                  scale=alt.Scale(domain=["Actual", "Forecast", "Same as last week"], range=[INK, ACCENT, MUTED]))
next_week = pd.DataFrame({"start": [week_start - pd.Timedelta(hours=12)], "end": [d["date"].max() + pd.Timedelta(hours=12)]})
band = alt.Chart(next_week).mark_rect(opacity=0.08, color=ACCENT).encode(x="start:T", x2="end:T")
band_label = alt.Chart(next_week).mark_text(align="left", dx=4, dy=8, baseline="top", fontSize=11, color=MUTED).encode(
    x="start:T", y=alt.value(0), text=alt.value("Next 7 days"))
lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
    y=alt.Y("transactions:Q", title="Transactions per day", scale=alt.Scale(zero=False)),
    color=color,
    strokeDash=alt.StrokeDash("series:N", legend=None,
                              scale=alt.Scale(domain=["Actual", "Forecast", "Same as last week"], range=[[1, 0], [1, 0], [4, 3]])),
    tooltip=[alt.Tooltip("date:T", format="%a %d %b"), "series:N", alt.Tooltip("transactions:Q", format=",.0f"),
             alt.Tooltip("rain_mm:Q", title="rain (mm)"), alt.Tooltip("public_holiday:N", title="public holiday")],
)
st.altair_chart((band + band_label + lines).properties(height=300), use_container_width=True)
st.caption("Past weeks show what the forecast said 1 to 7 days before, against what happened.")
source("daily_sales.csv")

# ---------------------------------------------------------------- staffing heatmap
st.subheader("Staff needed by hour, next 7 days")
s = pick(staffing).groupby(["date", "hour"], as_index=False)[["forecast_orders", "suggested_staff", "scheduled_staff"]].sum()
s = s[s["suggested_staff"] + s["scheduled_staff"] > 0]
s["day"] = s["date"].dt.strftime("%a")  # one week, so the weekday is enough; the tooltip has the date
s["vs_roster"] = s["scheduled_staff"] - s["suggested_staff"]
day_order = sorted(s["day"].unique(), key=lambda x: s.loc[s["day"] == x, "date"].iloc[0])
base = alt.Chart(s).encode(
    x=alt.X("day:O", title=None, sort=day_order, axis=alt.Axis(labelAngle=0, orient="top", labelOverlap=False)),
    y=alt.Y("hour:O", title="Hour starting", axis=alt.Axis(labelOverlap=False)),
    tooltip=[alt.Tooltip("date:T", title="day", format="%a %d %b"), "hour:O", alt.Tooltip("forecast_orders:Q", title="forecast transactions", format=",.0f"),
             alt.Tooltip("suggested_staff:Q", title="suggested staff"), alt.Tooltip("scheduled_staff:Q", title="rostered"),
             alt.Tooltip("vs_roster:Q", title="rostered minus suggested")],
)
heat = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
    color=alt.Color("suggested_staff:Q", title="Staff", scale=alt.Scale(scheme="blues"), legend=None))
labels = base.mark_text(fontSize=11).encode(
    text="suggested_staff:Q",
    color=alt.condition(alt.datum.suggested_staff > s["suggested_staff"].quantile(0.6), alt.value("white"), alt.value("#0b0b0b")))
st.altair_chart((heat + labels).properties(height=26 * s["hour"].nunique()), use_container_width=True)
total_s, total_r = int(s["suggested_staff"].sum()), int(s["scheduled_staff"].sum())
st.markdown(f"Suggested **{total_s:,}** staff-hours vs **{total_r:,}** on the current roster "
            f"({total_r - total_s:+,} rostered vs suggested). Hover a cell for the hour's detail.")
source("staffing_plan.csv")

# ---------------------------------------------------------------- orders
st.subheader("Suggested orders")
o = pick(orders).copy()
show = o[["outlet", "item", "unit", "forecast_demand", "on_hand", "suggested_order"]].rename(columns={
    "outlet": "Outlet", "item": "Item", "unit": "Unit", "forecast_demand": "Forecast use",
    "on_hand": "On hand", "suggested_order": "Order"})
if outlet != ALL:
    show = show.drop(columns="Outlet")
st.dataframe(show, hide_index=True, use_container_width=True,
             column_config={"Forecast use": st.column_config.NumberColumn(format="%.1f"),
                            "On hand": st.column_config.NumberColumn(format="%.1f"),
                            "Order": st.column_config.NumberColumn(format="%.0f")})
st.caption("Order = forecast use × (1 + buffer) − usable stock, rounded up to whole packs. Fresh items cover "
           "tomorrow and leftovers aren't counted; dry goods cover the days until the next delivery.")
source("suggested_orders.csv")

# ---------------------------------------------------------------- waste and gaps
st.subheader(f"Last {client.review_days} days by outlet")
w = summary.copy()
bars = alt.Chart(w).mark_bar(color=ACCENT, cornerRadiusEnd=4, height={"band": 0.5}).encode(
    x=alt.X("waste_sgd:Q", title="Waste (S$)", axis=alt.Axis(format="$,.0f")),
    y=alt.Y("outlet:N", title=None, sort="-x"),
    opacity=alt.condition(alt.datum.outlet == outlet, alt.value(1), alt.value(0.85 if outlet == ALL else 0.35)),
    tooltip=["outlet:N", alt.Tooltip("waste_sgd:Q", title="waste (S$)", format="$,.0f")],
)
bar_labels = bars.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text("waste_sgd:Q", format="$,.0f"), color=alt.value(INK))
st.altair_chart((bars + bar_labels).properties(height=200), use_container_width=True)
st.dataframe(
    w.rename(columns={"outlet": "Outlet", "waste_sgd": "Waste (S$)", "overstaffed_hours": "Over-staffed staff-hours",
                      "understaffed_hours": "Under-staffed staff-hours", "stockouts": "Stock-outs (item-days)"}),
    hide_index=True, use_container_width=True, column_config={"Waste (S$)": st.column_config.NumberColumn(format="dollar")})
source("outlet_summary.csv")

# ---------------------------------------------------------------- savings estimate
row = savings[savings["outlet"] == outlet].iloc[0]
with st.expander(f"Savings estimate: about S${row['estimated_saving_sgd_per_month']:,.0f} a month (estimate, not a guarantee)"):
    st.markdown(
        f"- Waste: {row['avoidable_waste_share']:.0%} of the last {client.review_days} days' waste is assumed avoidable "
        f"→ **S${row['avoidable_waste_sgd_per_month']:,.0f}/month**\n"
        f"- Staffing: {row['recoverable_overstaff_share']:.0%} of over-staffed hours assumed cut, at "
        f"S${row['hourly_wage_sgd']:.2f}/hour → **S${row['recoverable_wages_sgd_per_month']:,.0f}/month**\n\n"
        "These assumptions are set per client and agreed with you. Under-staffed hours aren't counted here, "
        "but they cost sales and service."
    )
    source("savings_estimate.csv")
