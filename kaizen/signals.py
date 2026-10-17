"""Outside signals: public holidays, school holidays and weather.

Each source has its own function and its own cached file in data/signals/.
Network calls happen only here, in `make signals` (which `make forecast` runs
first). The step writes one table per client, `<outputs>/signals_daily.csv`,
and the forecast reads only that file, so training never touches the internet.

Sources
  Public holidays  data.gov.sg, "Singapore Public Holidays" (Ministry of Manpower)
  School holidays  data/signals/school_holidays.yaml, hand-maintained from MOE
  Weather          Meteorological Service Singapore historical daily records
                   (weather.gov.sg), one station. Days without a record (not yet
                   published, or the forecast week) fall back to the client's
                   weather.csv if present, otherwise to the monthly average of
                   the recorded days. Every row says which source it came from.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

from kaizen.config import HORIZON_DAYS, SIGNALS_DIR, load_client, setup_logging
from kaizen.contract import load_inputs

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "kaizen-labs-pipeline/0.1 (+https://kaizenlabs.sg)"}
HOLIDAY_COLLECTION_URL = "https://api-production.data.gov.sg/v2/public/api/collections/691/metadata"
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"
MSS_URL = "https://www.weather.gov.sg/files/dailydata/DAILYDATA_{station}_{yyyymm}.csv"


def _get(url: str, params: dict | None = None, tries: int = 3) -> requests.Response:
    for attempt in range(tries):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 429:  # data.gov.sg rate-limits bursts of calls
            return resp
        time.sleep(5 * (attempt + 1))
    return resp


# ---------------------------------------------------------------- public holidays


def fetch_public_holidays() -> pd.DataFrame:
    """Download every year in the data.gov.sg public holiday collection."""
    meta = _get(HOLIDAY_COLLECTION_URL)
    meta.raise_for_status()
    rows = []
    for dataset_id in meta.json()["data"]["collectionMetadata"]["childDatasets"]:
        resp = _get(DATASTORE_URL, params={"resource_id": dataset_id, "limit": 100})
        resp.raise_for_status()
        for rec in resp.json()["result"]["records"]:
            rows.append({"date": rec["date"], "holiday": rec["holiday"].replace("’", "'"), "dataset": dataset_id})
        time.sleep(1)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def public_holidays(start: pd.Timestamp, end: pd.Timestamp, signals_dir: Path = SIGNALS_DIR,
                    allow_network: bool = True) -> pd.DataFrame:
    """Public holidays between start and end, from cache, refreshed if a year is missing."""
    cache = signals_dir / "public_holidays.csv"
    df = pd.read_csv(cache, parse_dates=["date"]) if cache.exists() else pd.DataFrame(columns=["date", "holiday", "dataset"])
    needed = set(range(start.year, end.year + 1))
    missing = needed - set(pd.to_datetime(df["date"]).dt.year)
    if missing and allow_network:
        try:
            df = fetch_public_holidays()
            df.to_csv(cache, index=False, date_format="%Y-%m-%d")
            log.info("Public holidays refreshed from data.gov.sg (%d dates) -> %s", len(df), cache)
        except requests.RequestException as exc:
            log.warning("Could not reach data.gov.sg for public holidays (%s); using cache", exc)
        missing = needed - set(pd.to_datetime(df["date"]).dt.year)
    if missing:
        log.warning("No public holiday list for %s; those years are treated as having none", sorted(missing))
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)][["date", "holiday"]].reset_index(drop=True)


# ---------------------------------------------------------------- school holidays


def school_holiday_dates(start: pd.Timestamp, end: pd.Timestamp, signals_dir: Path = SIGNALS_DIR) -> pd.DatetimeIndex:
    spec = yaml.safe_load((signals_dir / "school_holidays.yaml").read_text())
    uncovered = set(range(start.year, end.year + 1)) - set(spec["years_covered"])
    if uncovered:
        log.warning("school_holidays.yaml has no MOE term breaks for %s; add them from moe.gov.sg", sorted(uncovered))
    days = [pd.date_range(b["start"], b["end"], freq="D") for b in spec["breaks"]]
    all_days = days[0].append(days[1:]) if days else pd.DatetimeIndex([])
    return all_days[(all_days >= start) & (all_days <= end)].unique().sort_values()


# ---------------------------------------------------------------- weather


def fetch_mss_month(station: str, year: int, month: int) -> pd.DataFrame | None:
    """One month of MSS daily records, or None if that month is not published."""
    resp = _get(MSS_URL.format(station=station, yyyymm=f"{year}{month:02d}"))
    if resp.status_code == 404 or not resp.content.lstrip().startswith((b"\xef\xbb\xbfStation", b"Station")):
        return None
    resp.raise_for_status()
    try:
        text = resp.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = resp.content.decode("cp1252")
    raw = pd.read_csv(io.StringIO(text))
    rain_col = next(c for c in raw.columns if c.startswith("Daily Rainfall Total"))
    temp_col = next(c for c in raw.columns if c.startswith("Maximum Temperature"))
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[["Year", "Month", "Day"]]),
        "rain_mm": pd.to_numeric(raw[rain_col], errors="coerce"),  # MSS marks gaps with a dash
        "max_temp_c": pd.to_numeric(raw[temp_col], errors="coerce"),
    })
    return out.dropna(subset=["rain_mm", "max_temp_c"], how="all")


def mss_weather(start: pd.Timestamp, end: pd.Timestamp, station: str, signals_dir: Path = SIGNALS_DIR,
                allow_network: bool = True) -> pd.DataFrame:
    """MSS daily records for the period, fetching months not yet in the cache."""
    cache = signals_dir / f"weather_mss_{station}.csv"
    df = pd.read_csv(cache, parse_dates=["date"]) if cache.exists() else pd.DataFrame(columns=["date", "rain_mm", "max_temp_c"])
    df["date"] = pd.to_datetime(df["date"])
    # A month counts as cached only when every day is there; MSS publishes with a lag,
    # so a partly published month is fetched again next time.
    per_month = df["date"].dt.to_period("M").value_counts()
    complete = {m for m, n in per_month.items() if n == m.days_in_month}
    last_possible = pd.Timestamp.today().to_period("M")
    wanted = [m for m in pd.period_range(start, end, freq="M") if m <= last_possible and m not in complete]
    if allow_network and wanted:
        fetched = []
        for i, month in enumerate(wanted):
            try:
                part = fetch_mss_month(station, month.year, month.month)
            except requests.RequestException as exc:
                log.warning("MSS weather unreachable (%s); stopping fetch", exc)
                break
            if part is None:
                log.info("MSS has not published %s for station %s yet", month, station)
                continue
            fetched.append(part)
            if i:
                time.sleep(0.5)
        if fetched:
            df = pd.concat([df, *fetched]).drop_duplicates("date", keep="last").sort_values("date")
            df.to_csv(cache, index=False, date_format="%Y-%m-%d")
            log.info("MSS weather: cached %d new month(s) -> %s", len(fetched), cache)
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def weather(start: pd.Timestamp, end: pd.Timestamp, station: str = "S24", fallback: pd.DataFrame | None = None,
            fallback_label: str = "client_weather_csv", signals_dir: Path = SIGNALS_DIR, allow_network: bool = True) -> pd.DataFrame:
    """Daily rain_mm and max_temp_c for every day in the period, with a `weather_source` column."""
    days = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    try:
        mss = mss_weather(start, end, station, signals_dir, allow_network)
    except (requests.RequestException, OSError, ValueError) as exc:
        log.warning("MSS weather unavailable (%s)", exc)
        mss = pd.DataFrame(columns=["date", "rain_mm", "max_temp_c"])
    out = days.merge(mss.assign(weather_source="mss"), on="date", how="left")
    out[["rain_mm", "max_temp_c"]] = out[["rain_mm", "max_temp_c"]].astype(float)

    if fallback is not None:
        fb = fallback[["date", "rain_mm", "max_temp_c"]].rename(columns={"rain_mm": "fb_rain", "max_temp_c": "fb_temp"})
        out = out.merge(fb, on="date", how="left")
        use_fb = out["weather_source"].isna() & out["fb_rain"].notna()
        out.loc[use_fb, ["rain_mm", "max_temp_c"]] = out.loc[use_fb, ["fb_rain", "fb_temp"]].to_numpy()
        out.loc[use_fb, "weather_source"] = fallback_label
        out = out.drop(columns=["fb_rain", "fb_temp"])

    gap = out["weather_source"].isna()
    if gap.any():
        known = out[~gap]
        if known.empty:
            raise RuntimeError("No weather available from MSS or a weather.csv; cannot build signals")
        clim = known.groupby(known["date"].dt.month)[["rain_mm", "max_temp_c"]].mean()
        overall = known[["rain_mm", "max_temp_c"]].mean()
        month = out.loc[gap, "date"].dt.month
        for col in ("rain_mm", "max_temp_c"):
            out.loc[gap, col] = month.map(clim[col]).fillna(overall[col]).to_numpy()
        out.loc[gap, "weather_source"] = "monthly_average"

    counts = out["weather_source"].value_counts().to_dict()
    if counts.get("mss", 0) < len(out):
        log.warning("Weather sources used: %s (MSS records cover %d of %d days)", counts, counts.get("mss", 0), len(out))
    else:
        log.info("Weather: all %d days from MSS station %s", len(out), station)
    return out


# ---------------------------------------------------------------- combined table


def daily_signals(start: pd.Timestamp, end: pd.Timestamp, station: str = "S24", fallback_weather: pd.DataFrame | None = None,
                  signals_dir: Path = SIGNALS_DIR, allow_network: bool = True) -> pd.DataFrame:
    """One row per day: holiday flags and weather. This is what the forecast reads."""
    out = weather(start, end, station, fallback_weather, signals_dir=signals_dir, allow_network=allow_network)
    ph = public_holidays(start, end, signals_dir, allow_network)
    out = out.merge(ph.rename(columns={"holiday": "public_holiday"}), on="date", how="left")
    out["public_holiday"] = out["public_holiday"].fillna("")
    out["is_public_holiday"] = (out["public_holiday"] != "").astype(int)
    out["is_school_holiday"] = out["date"].isin(school_holiday_dates(start, end, signals_dir)).astype(int)
    out["rain_mm"] = out["rain_mm"].round(1)
    out["max_temp_c"] = out["max_temp_c"].round(1)
    cols = ["date", "public_holiday", "is_public_holiday", "is_school_holiday", "rain_mm", "max_temp_c", "weather_source"]
    return out[cols]


def signals_path(outputs_dir: Path) -> Path:
    return outputs_dir / "signals_daily.csv"


def load_signals(outputs_dir: Path) -> pd.DataFrame:
    path = signals_path(outputs_dir)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make signals` first.")
    df = pd.read_csv(path, parse_dates=["date"], keep_default_na=False)
    return df


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch and cache outside signals for a client's date range.")
    parser.add_argument("--client", default="demo")
    parser.add_argument("--offline", action="store_true", help="use cached files only")
    args = parser.parse_args()
    client = load_client(args.client)
    inputs = load_inputs(client.inputs_dir)
    start = inputs.sales["timestamp"].min().normalize()
    # Two extra weeks past the forecast window so "days to next holiday" is right at its end.
    end = inputs.history_end + pd.Timedelta(days=HORIZON_DAYS + 14)
    table = daily_signals(start, end, client.weather_station, inputs.weather, allow_network=not args.offline)
    client.outputs_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(signals_path(client.outputs_dir), index=False, date_format="%Y-%m-%d")
    print(f"Signals {start:%Y-%m-%d} to {end:%Y-%m-%d}: {table['is_public_holiday'].sum()} public holidays, "
          f"{table['is_school_holiday'].sum()} school-holiday days -> {signals_path(client.outputs_dir)}")


if __name__ == "__main__":
    main()
