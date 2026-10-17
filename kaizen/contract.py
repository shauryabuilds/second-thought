"""The data contract: the files a client provides, and the checks they must pass.

Only `sales.csv` is required. Everything else improves the plan when present.

    sales.csv   timestamp (hourly), outlet, transactions
    items.csv   date, outlet, item, units_sold                 (optional)
    stock.csv   date, outlet, item, unit, on_hand              (optional)
    roster.csv  date, outlet, hour, scheduled_staff            (optional)
    waste.csv   date, outlet, waste_sgd                        (optional)
    menu.csv    item, unit, usage_per_transaction              (optional, set up with the client)
    weather.csv date, rain_mm, max_temp_c                      (optional, fallback weather)

Files must contain exactly these columns, no more. That is deliberate: the
pipeline never needs anything about customers, so a stray column (a name, a
phone number, a card number) is rejected rather than carried along.

Run `python -m kaizen.contract --client <key>` to validate a client's inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from kaizen.config import load_client, setup_logging

log = logging.getLogger(__name__)


class ContractError(ValueError):
    """An input file does not match the data contract."""


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SalesRow(_Row):
    timestamp: dt.datetime
    outlet: str = Field(min_length=1)
    transactions: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def on_the_hour(cls, v: dt.datetime) -> dt.datetime:
        if v.tzinfo is not None:
            raise ValueError("use local Singapore time without a timezone offset")
        if (v.minute, v.second, v.microsecond) != (0, 0, 0):
            raise ValueError("timestamps must be on the hour (start of the hour bucket)")
        return v


class ItemsRow(_Row):
    date: dt.date
    outlet: str = Field(min_length=1)
    item: str = Field(min_length=1)
    units_sold: float = Field(ge=0)


class StockRow(_Row):
    date: dt.date
    outlet: str = Field(min_length=1)
    item: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    on_hand: float = Field(ge=0)


class RosterRow(_Row):
    date: dt.date
    outlet: str = Field(min_length=1)
    hour: int = Field(ge=0, le=23)
    scheduled_staff: int = Field(ge=0)


class WasteRow(_Row):
    date: dt.date
    outlet: str = Field(min_length=1)
    waste_sgd: float = Field(ge=0)


class MenuRow(_Row):
    item: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    usage_per_transaction: float = Field(gt=0)


class WeatherRow(_Row):
    date: dt.date
    rain_mm: float = Field(ge=0)
    max_temp_c: float = Field(ge=15, le=45)


@dataclass(frozen=True)
class FileSpec:
    name: str
    row: type[_Row]
    key: tuple[str, ...]  # columns that must be unique together
    required: bool = False

    @property
    def columns(self) -> list[str]:
        return list(self.row.model_fields)


SPECS: dict[str, FileSpec] = {
    s.name: s
    for s in [
        FileSpec("sales", SalesRow, ("timestamp", "outlet"), required=True),
        FileSpec("items", ItemsRow, ("date", "outlet", "item")),
        FileSpec("stock", StockRow, ("date", "outlet", "item")),
        FileSpec("roster", RosterRow, ("date", "outlet", "hour")),
        FileSpec("waste", WasteRow, ("date", "outlet")),
        FileSpec("menu", MenuRow, ("item",)),
        FileSpec("weather", WeatherRow, ("date",)),
    ]
}

MAX_ERRORS_SHOWN = 10


def validate_frame(df: pd.DataFrame, spec: FileSpec, source: str = "") -> pd.DataFrame:
    """Check a frame against its spec and return it with proper dtypes.

    Raises ContractError listing the first few problems with their CSV line numbers.
    """
    where = source or f"{spec.name}.csv"
    expected = spec.columns
    if sorted(df.columns) != sorted(expected):
        missing = sorted(set(expected) - set(df.columns))
        extra = sorted(set(df.columns) - set(expected))
        raise ContractError(
            f"{where}: columns must be exactly {expected}. Missing: {missing or 'none'}. Unexpected: {extra or 'none'}."
        )
    df = df[expected]
    if df.empty:
        raise ContractError(f"{where}: file has no rows")

    records = df.astype(object).where(df.notna(), None).to_dict("records")
    try:
        TypeAdapter(list[spec.row]).validate_python(records)
    except ValidationError as exc:
        lines = []
        for err in exc.errors()[:MAX_ERRORS_SHOWN]:
            row_idx, *field = err["loc"]
            lines.append(f"  line {int(row_idx) + 2}, {'.'.join(map(str, field))}: {err['msg']}")
        more = exc.error_count() - len(lines)
        raise ContractError(
            f"{where}: {exc.error_count()} invalid value(s)\n" + "\n".join(lines) + (f"\n  ... and {more} more" if more > 0 else "")
        ) from None

    # Values are valid; now give the frame the dtypes the rest of the pipeline expects.
    out = df.copy()
    for col, field in spec.row.model_fields.items():
        if field.annotation in (dt.datetime, dt.date):
            out[col] = pd.to_datetime(out[col])
        elif field.annotation is int:
            out[col] = pd.to_numeric(out[col]).astype("int64")
        elif field.annotation is float:
            out[col] = pd.to_numeric(out[col]).astype("float64")
        else:
            out[col] = out[col].astype(str).str.strip()

    dupes = out.duplicated(list(spec.key), keep=False)
    if dupes.any():
        first = out[dupes].head(3).to_dict("records")
        raise ContractError(f"{where}: {int(dupes.sum())} rows repeat the same {list(spec.key)}, e.g. {first}")
    return out.sort_values(list(spec.key)).reset_index(drop=True)


@dataclass
class Inputs:
    sales: pd.DataFrame
    items: pd.DataFrame | None = None
    stock: pd.DataFrame | None = None
    roster: pd.DataFrame | None = None
    waste: pd.DataFrame | None = None
    menu: pd.DataFrame | None = None
    weather: pd.DataFrame | None = None

    @property
    def outlets(self) -> list[str]:
        return sorted(self.sales["outlet"].unique())

    @property
    def history_end(self) -> pd.Timestamp:
        """The last full day of sales history."""
        return self.sales["timestamp"].max().normalize()


def read_file(path: Path, spec: FileSpec) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return validate_frame(df, spec, source=str(path))


def check_outlets(inputs: Inputs) -> None:
    known = set(inputs.outlets)
    for name in ("items", "stock", "roster", "waste"):
        df = getattr(inputs, name)
        if df is not None:
            unknown = sorted(set(df["outlet"]) - known)
            if unknown:
                raise ContractError(f"{name}.csv: outlets not found in sales.csv: {unknown}")


def load_inputs(inputs_dir: Path) -> Inputs:
    """Read and validate every contract file present in `inputs_dir`."""
    frames: dict[str, pd.DataFrame] = {}
    for name, spec in SPECS.items():
        path = inputs_dir / f"{name}.csv"
        if path.exists():
            frames[name] = read_file(path, spec)
        elif spec.required:
            raise ContractError(f"Required file missing: {path}")
    inputs = Inputs(**frames)
    check_outlets(inputs)
    return inputs


def describe(inputs: Inputs) -> str:
    s = inputs.sales
    lines = [
        f"sales.csv   {len(s):>8,} rows  {s['timestamp'].min():%Y-%m-%d} to {s['timestamp'].max():%Y-%m-%d}  "
        f"{len(inputs.outlets)} outlets: {', '.join(inputs.outlets)}"
    ]
    for name in SPECS:
        if name == "sales":
            continue
        df = getattr(inputs, name)
        lines.append(f"{name + '.csv':<11} {'(not provided)' if df is None else f'{len(df):>8,} rows'}")
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Validate a client's input files against the data contract.")
    parser.add_argument("--client", default="demo")
    args = parser.parse_args()
    client = load_client(args.client)
    try:
        inputs = load_inputs(client.inputs_dir)
    except ContractError as exc:
        raise SystemExit(f"Contract check FAILED\n{exc}")
    print(f"Contract check passed for {client.name} ({client.inputs_dir})\n{describe(inputs)}")


if __name__ == "__main__":
    main()
