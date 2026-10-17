"""One-page PDF of next week's plan for a client: `make report CLIENT=<key>`.

Built from the same output files as the dashboard, so the numbers match:
customers per day, people per shift, menu items, ingredient use and the order list.
Writes <outputs>/<client key>_summary.pdf.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from kaizen.config import ClientConfig, load_client, setup_logging  # noqa: E402

ACCENT, INK, MUTED, LINE = "#1D4ED8", "#111418", "#6b6a66", "#e4e3df"
STAFF_FILL = {3: "#ffffff", 4: "#e3eefc", 5: "#b7d3f6", 6: "#86b6ef"}
WHOLE_UNITS = {"pcs", "slices", "units"}
ROW = 0.0142  # height of one table row, as a share of the page
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.parse_math": False})  # "S$" is money, not maths


def qty(value: float, unit: str) -> str:
    return f"{value:,.0f}" if unit in WHOLE_UNITS else f"{value:,.1f}"


def short(outlet: str) -> str:
    return outlet.split(" - ")[0]


class Page:
    """Stacks headings and tables down an A4 page."""

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.y = 0.965
        self.left, self.width = 0.06, 0.88

    def text(self, s: str, size: float = 7, color: str = INK, weight: str = "normal", gap: float = 0.018, **kw) -> None:
        self.fig.text(self.left, self.y, s, fontsize=size, color=color, weight=weight, va="top", **kw)
        self.y -= gap

    def heading(self, s: str) -> None:
        self.y -= 0.008
        self.text(s, size=9.5, weight="bold", gap=0.02)

    def table(self, df: pd.DataFrame, first_col: float = 0.26, fills: dict | None = None) -> None:
        height = ROW * (len(df) + 1)
        ax = self.fig.add_axes([self.left, self.y - height, self.width, height])
        ax.axis("off")
        n = len(df.columns)
        widths = [first_col] + [(1 - first_col) / (n - 1)] * (n - 1)
        t = ax.table(cellText=df.values, colLabels=df.columns, colWidths=widths, bbox=[0, 0, 1, 1], cellLoc="right")
        t.auto_set_font_size(False)
        t.set_fontsize(6.8)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(LINE)
            cell.set_linewidth(0.5)
            cell.PAD = 0.04
            if c == 0:
                cell._loc = "left"
                cell.set_text_props(ha="left")
            if r == 0:
                cell.set_text_props(color=MUTED, weight="bold")
            elif fills and (r - 1, c) in fills:
                cell.set_facecolor(fills[(r - 1, c)])
        self.y -= height + 0.012


def build(client: ClientConfig) -> str:
    out = client.outputs_dir
    daily = pd.read_csv(out / "daily_sales.csv", parse_dates=["date"], keep_default_na=False, na_values=[""])
    shifts = pd.read_csv(out / "staffing_plan.csv", parse_dates=["date"])
    skus = pd.read_csv(out / "sku_forecast.csv", parse_dates=["date"])
    ingredients = pd.read_csv(out / "ingredient_forecast.csv", parse_dates=["date"])
    orders = pd.read_csv(out / "suggested_orders.csv")
    acc = pd.read_csv(out / "backtest_accuracy.csv")
    savings = pd.read_csv(out / "savings_estimate.csv")

    start = shifts["date"].min()
    days = pd.date_range(start, periods=7, freq="D")
    labels = [f"{d:%a %d}" for d in days]
    p = Page()

    # Header: the logo (ink with an accent dot), then the week.
    x = p.left
    for part, colour in (("kaizen", INK), (".", ACCENT), ("labs", INK)):
        t = p.fig.text(x, p.y, part, fontsize=11, family="monospace", weight="bold", color=colour, va="top")
        x += t.get_window_extent(p.fig.canvas.get_renderer()).width / p.fig.bbox.width
    p.y -= 0.027
    p.text(f"{client.name}: next week, {start:%a %d} to {days[-1]:%a %d %b %Y}", size=14, weight="bold", gap=0.022)
    if client.is_sample:
        p.text("Sample data for a fictional business.", color=MUTED, gap=0.012)

    # 1. Customers per day
    week = daily[daily["date"] >= start]
    c = week.pivot_table(index="outlet", columns="date", values="forecast", aggfunc="sum")
    c.loc["All outlets"] = c.sum()
    lw = week.groupby("outlet")["last_week_rule"].sum()
    lw.loc["All outlets"] = lw.sum()
    tbl = pd.DataFrame({"Outlet": c.index})
    for d, lab in zip(days, labels):
        tbl[lab] = c[d].map("{:,.0f}".format).to_numpy()
    tbl["Week"] = c.sum(axis=1).map("{:,.0f}".format).to_numpy()
    tbl["vs last wk"] = (c.sum(axis=1) / lw - 1).map("{:+.0%}".format).to_numpy()
    p.heading("1. Customers expected")
    p.table(tbl)
    holidays = week.loc[week["public_holiday"].fillna("") != "", ["date", "public_holiday"]].drop_duplicates("date")
    for _, r in holidays.iterrows():
        p.text(f"{r['date']:%a %d %b} is {r['public_holiday']}; already in the forecast.", color=MUTED, gap=0.014)

    # 2. People per shift
    s = shifts.copy()
    s["row"] = s["outlet"].map(short) + " · " + s["shift"]
    s["cell"] = s["suggested_staff"].astype(str).where(s["suggested_staff"] > 0, "") + \
        (s["needed_at_peak"] > s["suggested_staff"]).map({True: "+", False: ""})
    order = [f"{short(o)} · {x.name}" for o in sorted(s["outlet"].unique()) for x in client.shifts]
    grid = s.pivot_table(index="row", columns="date", values="cell", aggfunc="first").reindex([r for r in order if r in set(s["row"])])
    people = s.pivot_table(index="row", columns="date", values="suggested_staff", aggfunc="first").reindex(grid.index)
    tbl = pd.DataFrame({"Outlet · shift": grid.index})
    for d, lab in zip(days, labels):
        tbl[lab] = grid[d].fillna("").to_numpy() if d in grid else ""
    fills = {(i, j + 1): STAFF_FILL.get(int(v), "#ffffff") for i, row in enumerate(people.reindex(columns=days).to_numpy())
             for j, v in enumerate(row) if pd.notna(v) and v > 0}
    hrs = ", ".join(f"{x.name} {x.start:02d}-{x.end % 24:02d}" for x in client.shifts)
    p.heading("2. People per shift")
    p.table(tbl, fills=fills)
    p.text(f"Shifts: {hrs}. People = enough for the shift's busiest hour at {client.staffing.transactions_per_staff_hour:.0f} "
           f"customers per person per hour, {client.staffing.min_staff} to {client.staffing.max_staff}.", color=MUTED, gap=0.013)
    p.text("+ = the busiest hour needs more than the maximum. Blank = closed.", color=MUTED, gap=0.014)
    if shifts["scheduled_staff"].notna().any():
        diff = shifts[shifts["suggested_staff"] != shifts["scheduled_staff"]]
        p.text(f"Changes to the current roster: {len(diff)} shifts "
               f"({int((diff['suggested_staff'] > diff['scheduled_staff']).sum())} need more people, "
               f"{int((diff['suggested_staff'] < diff['scheduled_staff']).sum())} fewer).", color=MUTED, gap=0.014)

    # 3. Menu items, all outlets
    k = skus.pivot_table(index="item", columns="date", values="forecast_units", aggfunc="sum")
    k = k.loc[k.sum(axis=1).sort_values(ascending=False).index]
    tbl = pd.DataFrame({"Menu item": k.index})
    for d, lab in zip(days, labels):
        tbl[lab] = k[d].map("{:,.0f}".format).to_numpy()
    tbl["Week"] = k.sum(axis=1).map("{:,.0f}".format).to_numpy()
    p.heading("3. What will sell, all outlets")
    p.table(tbl)

    # 4. Ingredients, all outlets, with the total order
    g = ingredients.pivot_table(index=["ingredient", "unit"], columns="date", values="forecast_qty", aggfunc="sum")
    ordered = orders.groupby(["item", "unit"])["suggested_order"].sum()
    tbl = pd.DataFrame({"Ingredient": [f"{i} ({u})" for i, u in g.index]})
    for d, lab in zip(days, labels):
        tbl[lab] = [qty(v, u) for v, u in zip(g[d], g.index.get_level_values("unit"))]
    tbl["Order now"] = [qty(ordered.get(ix, 0.0), ix[1]) for ix in g.index]
    p.heading("4. Ingredients used per day, and what to order now (all outlets)")
    p.table(tbl)
    p.text("Order = forecast use until the next delivery + buffer - stock on hand, in whole packs; per outlet in "
           "suggested_orders.csv.", color=MUTED, gap=0.02)

    # Footer: reliability and the (labelled) savings estimate.
    a = acc[(acc["outlet"] == "All outlets") & (acc["level"] == "day")].set_index("method")["accuracy"]
    total = savings[savings["outlet"] == "All outlets"].iloc[0]
    p.text(f"Forecast accuracy, daily customers, a week ahead (last 8 weeks): {a['model']:.0%}, vs {a['last_week_rule']:.0%} "
           "for \"same as last week\".", gap=0.014)
    p.text(f"Savings estimate (not a guarantee): about S${total['estimated_saving_sgd_per_month']:,.0f} a month, assuming "
           f"{total['avoidable_waste_share']:.0%} of waste avoided and {total['recoverable_overstaff_share']:.0%} of "
           f"over-staffed hours cut at S${total['hourly_wage_sgd']:.2f}/hour.", gap=0.02)
    p.fig.text(p.left, 0.016, f"Every figure comes from the files in {out.relative_to(out.parents[1])}/ · kaizenlabs.sg · "
               "shaurya@kaizenlabs.sg", fontsize=6.5, color=MUTED)

    path = out / f"{client.key}_summary.pdf"
    p.fig.savefig(path)
    plt.close(p.fig)
    return str(path)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Export a one-page PDF of next week's plan.")
    parser.add_argument("--client", default="demo")
    print(f"Report -> {build(load_client(parser.parse_args().client))}")


if __name__ == "__main__":
    main()
