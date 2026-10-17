"""One-page PDF summary for a client: `make report CLIENT=<key>`.

Built from the same output files as the dashboard, so the numbers match.
Writes <outputs>/<client key>_summary.pdf.
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from kaizen.config import ClientConfig, load_client, setup_logging  # noqa: E402

ACCENT, INK, MUTED, FAINT = "#1D4ED8", "#111418", "#6b6a66", "#a3a29c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": FAINT, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def table(ax, df: pd.DataFrame, col_widths: list[float]) -> None:
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, colWidths=col_widths, loc="upper left", cellLoc="right", colLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#e4e3df")
        cell.set_linewidth(0.5 if r else 0.8)
        cell.set_height(0.11 if len(df) < 8 else 0.07)
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if r == 0:
            cell.set_text_props(color=MUTED, weight="bold")


def build(client: ClientConfig) -> str:
    out = client.outputs_dir
    daily = pd.read_csv(out / "daily_sales.csv", parse_dates=["date"])
    acc = pd.read_csv(out / "backtest_accuracy.csv")
    staffing = pd.read_csv(out / "staffing_plan.csv", parse_dates=["date"])
    orders = pd.read_csv(out / "suggested_orders.csv")
    summary = pd.read_csv(out / "outlet_summary.csv")
    savings = pd.read_csv(out / "savings_estimate.csv")
    week = staffing["date"].min()

    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    x = 0.06
    for part, colour in (("kaizen", INK), (".", ACCENT), ("labs", INK)):  # the logo: ink with an accent dot
        t = fig.text(x, 0.965, part, fontsize=11, family="monospace", weight="bold", color=colour)
        x += t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    fig.text(0.06, 0.935, f"{client.name}: plan for the week of {week:%a %d %b %Y}", fontsize=14, weight="bold", color=INK)
    if client.is_sample:
        fig.text(0.06, 0.918, "Sample data for a fictional business.", fontsize=8, color=MUTED)

    # Accuracy by outlet
    a = acc[acc["level"] == "hour"].pivot(index="outlet", columns="method", values="accuracy")
    order = [o for o in a.index if o != "All outlets"] + ["All outlets"]
    a = a.loc[order, ["model", "last_week_rule", "four_week_avg"]]
    weeks = daily["actual"].notna().sum() // max(daily["outlet"].nunique(), 1) // 7
    fig.text(0.06, 0.89, f"Forecast accuracy, hour by hour (back-test, last {weeks} weeks, 7 days ahead)", fontsize=9.5, weight="bold", color=INK)
    acc_tbl = pd.DataFrame({"Outlet": a.index, "Forecast": (a["model"] * 100).map("{:.1f}%".format),
                            "Same as last week": (a["last_week_rule"] * 100).map("{:.1f}%".format),
                            "4-week average": (a["four_week_avg"] * 100).map("{:.1f}%".format)})
    table(fig.add_axes([0.06, 0.745, 0.88, 0.135]), acc_tbl, [0.4, 0.2, 0.2, 0.2])

    # Daily actual vs forecast, all outlets
    ax = fig.add_axes([0.09, 0.53, 0.85, 0.17])
    d = daily.groupby("date")[["actual", "forecast", "last_week_rule"]].sum(min_count=1)
    ax.plot(d.index, d["actual"], color=INK, lw=1.4, label="Actual")
    ax.plot(d.index, d["forecast"], color=ACCENT, lw=1.4, label="Forecast")
    ax.plot(d.index, d["last_week_rule"], color=FAINT, lw=1.2, ls="--", label="Same as last week")
    ax.axvspan(week - pd.Timedelta(hours=12), d.index.max() + pd.Timedelta(hours=12), color=ACCENT, alpha=0.07, lw=0)
    ax.text(week, ax.get_ylim()[1], " next 7 days", va="top", fontsize=7, color=MUTED)
    ax.set_ylabel("Transactions per day, all outlets")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5)
    ax.grid(axis="y", color="#eeede9", lw=0.6)
    fig.text(0.06, 0.725, "Daily transactions", fontsize=9.5, weight="bold", color=INK)

    # Staffing and orders
    s = staffing.groupby("outlet")[["suggested_staff", "scheduled_staff"]].sum()
    fig.text(0.06, 0.475, "Next 7 days", fontsize=9.5, weight="bold", color=INK)
    plan_tbl = pd.DataFrame({
        "Outlet": s.index, "Suggested staff-hours": s["suggested_staff"].map("{:,}".format),
        "Rostered staff-hours": s["scheduled_staff"].map("{:,}".format),
        "Order lines": orders.groupby("outlet")["suggested_order"].apply(lambda x: int((x > 0).sum())).reindex(s.index).fillna(0).astype(int),
    })
    table(fig.add_axes([0.06, 0.345, 0.88, 0.12]), plan_tbl, [0.4, 0.2, 0.2, 0.2])

    # Last 28 days
    fig.text(0.06, 0.315, f"Last {client.review_days} days", fontsize=9.5, weight="bold", color=INK)
    sum_tbl = pd.DataFrame({
        "Outlet": summary["outlet"], "Waste": summary["waste_sgd"].map("S${:,.0f}".format),
        "Over-staffed hrs": summary["overstaffed_hours"].map("{:,.0f}".format),
        "Under-staffed hrs": summary["understaffed_hours"].map("{:,.0f}".format),
        "Stock-outs": summary["stockouts"].map("{:,.0f}".format),
    })
    table(fig.add_axes([0.06, 0.185, 0.88, 0.12]), sum_tbl, [0.32, 0.17, 0.17, 0.17, 0.17])
    fig.text(0.06, 0.215, "Hours are staff-hours above or below what the transactions needed. "
             "Stock-outs are item-days that closed with nothing left.", fontsize=6.5, color=MUTED)

    # Savings estimate
    total = savings[savings["outlet"] == "All outlets"].iloc[0]
    fig.text(0.06, 0.15, "Savings estimate (an estimate, not a guarantee)", fontsize=9.5, weight="bold", color=INK)
    fig.text(0.06, 0.09, (
        f"About S${total['estimated_saving_sgd_per_month']:,.0f} a month: S${total['avoidable_waste_sgd_per_month']:,.0f} from waste "
        f"(assuming {total['avoidable_waste_share']:.0%} of it is avoidable) and S${total['recoverable_wages_sgd_per_month']:,.0f} from wages\n"
        f"(assuming {total['recoverable_overstaff_share']:.0%} of over-staffed hours are cut at S${total['hourly_wage_sgd']:.2f}/hour). "
        "The assumptions are agreed with you and can be changed.\n"
        "Under-staffed hours are not counted here, but they cost sales and service."
    ), fontsize=8, color=INK, linespacing=1.6)
    fig.text(0.06, 0.04, "Accuracy = 1 − (total absolute error ÷ total actual transactions). Every figure comes from the files in "
             f"{out.relative_to(out.parents[1])}/.\nkaizenlabs.sg · shaurya@kaizenlabs.sg", fontsize=6.5, color=MUTED, linespacing=1.6)

    path = out / f"{client.key}_summary.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Export a one-page PDF summary for a client.")
    parser.add_argument("--client", default="demo")
    print(f"Report -> {build(load_client(parser.parse_args().client))}")


if __name__ == "__main__":
    main()
