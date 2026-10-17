"""Turn a client's raw POS export into contract files (kaizen/contract.py).

Every POS system exports something different: other column names, 15-minute
buckets, one row per receipt line, voids mixed in, a timezone suffix. All of
that is described in a small YAML file per client, `clients/<key>/mapping.yaml`,
so a new export needs a new mapping, not new code. See the two examples in
clients/example_15min/ and clients/example_txn/.

Only the columns named in the mapping are ever read from the raw file. If an
export also carries customer names, phone numbers or card details, they are
never loaded.

    make ingest CLIENT=<key>
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from kaizen.config import ClientConfig, load_client, setup_logging
from kaizen.contract import SPECS, ContractError, describe, load_inputs

log = logging.getLogger(__name__)
LOCAL_TZ = "Asia/Singapore"


class SalesMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str  # file name or glob inside the client's raw folder
    layout: Literal["buckets", "transactions"]
    outlet: str
    timestamp: str | None = None  # one column holding date and time ...
    date: str | None = None  # ... or separate date and time columns
    time: str | None = None
    timestamp_format: str | None = None  # strptime format; pandas guesses if omitted
    count: str | None = None  # buckets: column with the number of transactions
    transaction_id: str | None = None  # transactions: receipt number (rows per receipt are de-duplicated)
    exclude: dict[str, list[str]] = {}  # drop rows whose column has one of these values, e.g. voids

    @model_validator(mode="after")
    def check(self) -> "SalesMapping":
        if not self.timestamp and not (self.date and self.time):
            raise ValueError("give either `timestamp`, or both `date` and `time`")
        if self.layout == "buckets" and not self.count:
            raise ValueError("layout 'buckets' needs `count`")
        return self


class FileMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str
    columns: dict[str, str]  # contract column -> raw column
    date_format: str | None = None


class Mapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outlets: dict[str, str] = {}  # raw outlet name -> name used in every output; empty keeps raw names
    sales: SalesMapping
    files: dict[Literal["items", "stock", "roster", "waste", "menu", "recipes", "weather"], FileMapping] = {}


def load_mapping(path: Path) -> Mapping:
    if not path.exists():
        raise SystemExit(f"No mapping at {path}. Copy one from clients/example_15min/mapping.yaml and adjust it.")
    return Mapping(**yaml.safe_load(path.read_text()))


def read_raw(raw_dir: Path, pattern: str, columns: list[str]) -> pd.DataFrame:
    paths = sorted(raw_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No raw file matching '{pattern}' in {raw_dir}")
    frames = []
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in columns if c not in header]
        if missing:
            raise SystemExit(f"{path.name}: columns {missing} not found. Columns in the file: {list(header)}")
        frames.append(pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False))
        log.info("Read %s (%d rows)", path.name, len(frames[-1]))
    return pd.concat(frames, ignore_index=True)


def map_outlets(values: pd.Series, outlets: dict[str, str]) -> pd.Series:
    values = values.str.strip()
    if not outlets:
        return values
    unknown = sorted(set(values) - set(outlets))
    if unknown:
        raise SystemExit(f"Outlets in the export but not in mapping.yaml `outlets`: {unknown}. Add them so none are silently dropped.")
    return values.map(outlets)


def to_local_time(text: pd.Series, fmt: str | None) -> pd.Series:
    """Parse timestamps; ones with a UTC offset are converted to Singapore time."""
    has_offset = text.str.strip().str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any()
    ts = pd.to_datetime(text, format=fmt or "mixed", utc=bool(has_offset))
    if has_offset:
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts


def drop_partial_last_day(sales: pd.DataFrame) -> pd.DataFrame:
    """Exports often stop mid-day. A partial last day would look like a slump, so drop it."""
    last_hour = sales.groupby(sales["timestamp"].dt.normalize())["timestamp"].max().dt.hour
    if len(last_hour) > 7 and last_hour.iloc[-1] < last_hour.iloc[:-1].median() - 1:
        day = last_hour.index[-1]
        log.warning("Last day %s looks incomplete (last sale at %02d:00); dropping it", day.date(), last_hour.iloc[-1])
        sales = sales[sales["timestamp"].dt.normalize() < day]
    return sales


def ingest_sales(m: SalesMapping, outlets: dict[str, str], raw_dir: Path) -> pd.DataFrame:
    cols = [c for c in (m.timestamp, m.date, m.time, m.outlet, m.count, m.transaction_id, *m.exclude) if c]
    df = read_raw(raw_dir, m.file, list(dict.fromkeys(cols)))
    for col, values in m.exclude.items():
        drop = df[col].str.strip().isin(values)
        log.info("Excluded %d rows where %s is one of %s", int(drop.sum()), col, values)
        df = df[~drop]

    text = df[m.timestamp] if m.timestamp else df[m.date].str.strip() + " " + df[m.time].str.strip()
    out = pd.DataFrame({"timestamp": to_local_time(text, m.timestamp_format).dt.floor("h"),
                        "outlet": map_outlets(df[m.outlet], outlets)})
    if m.layout == "buckets":
        out["transactions"] = pd.to_numeric(df[m.count].str.replace(",", ""), errors="raise")
        sales = out.groupby(["timestamp", "outlet"], as_index=False)["transactions"].sum()
    elif m.transaction_id:
        out["id"] = df[m.transaction_id].str.strip()
        sales = out.groupby(["timestamp", "outlet"], as_index=False)["id"].nunique().rename(columns={"id": "transactions"})
    else:
        sales = out.groupby(["timestamp", "outlet"], as_index=False).size().rename(columns={"size": "transactions"})
    sales["transactions"] = sales["transactions"].round().astype(int)
    return drop_partial_last_day(sales).sort_values(["timestamp", "outlet"]).reset_index(drop=True)


def ingest_file(name: str, m: FileMapping, outlets: dict[str, str], raw_dir: Path) -> pd.DataFrame:
    spec = SPECS[name]
    missing = set(spec.columns) - set(m.columns)
    if missing:
        raise SystemExit(f"mapping.yaml files.{name}.columns is missing {sorted(missing)}")
    df = read_raw(raw_dir, m.file, list(m.columns.values()))
    out = pd.DataFrame({target: df[source].str.strip() for target, source in m.columns.items()})
    for col, field in spec.row.model_fields.items():
        if field.annotation is dt.date:
            out[col] = pd.to_datetime(out[col], format=m.date_format).dt.strftime("%Y-%m-%d")
        elif field.annotation in (int, float):
            out[col] = out[col].str.replace(",", "")
    if "outlet" in out:
        out["outlet"] = map_outlets(out["outlet"], outlets)
    return out[spec.columns]


def ingest(client: ClientConfig) -> None:
    mapping = load_mapping(client.mapping_path)
    out_dir = client.inputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sales = ingest_sales(mapping.sales, mapping.outlets, client.raw_dir)
    sales.to_csv(out_dir / "sales.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    for name, file_mapping in mapping.files.items():
        ingest_file(name, file_mapping, mapping.outlets, client.raw_dir).to_csv(out_dir / f"{name}.csv", index=False)

    try:
        inputs = load_inputs(out_dir)
    except ContractError as exc:
        raise SystemExit(f"Ingested files do not pass the data contract:\n{exc}")
    print(f"Ingested {client.name} -> {out_dir} (contract check passed)\n{describe(inputs)}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Map a raw POS export to contract files using clients/<key>/mapping.yaml.")
    parser.add_argument("--client", required=True)
    args = parser.parse_args()
    ingest(load_client(args.client))


if __name__ == "__main__":
    main()
