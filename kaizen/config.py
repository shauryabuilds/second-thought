"""Client configuration: where a client's files live and the planning assumptions.

Each client has a folder `clients/<key>/` with:
  - client.yaml   paths and planning parameters (this module)
  - mapping.yaml  how to read their raw POS export (see kaizen/ingest.py), optional

Anything not set in client.yaml falls back to the defaults below, so a new client
needs only a name and an inputs/outputs folder to get started.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CLIENTS_DIR = ROOT / "clients"
SIGNALS_DIR = ROOT / "data" / "signals"

# The forecast always looks this many days past the last day of sales history.
HORIZON_DAYS = 7


class StaffingParams(BaseModel):
    transactions_per_staff_hour: float = Field(20.0, gt=0)
    min_staff: int = Field(2, ge=0)


class ItemParams(BaseModel):
    buffer: float = Field(0.10, ge=0)  # safety margin on top of forecast demand
    pack_size: float = Field(1.0, gt=0)  # orders are rounded up to whole packs
    cover_days: int = Field(7, ge=1, le=HORIZON_DAYS)  # days of demand one order covers
    perishable: bool = False  # leftovers are written off, so stock on hand is not counted


class SavingsAssumptions(BaseModel):
    hourly_wage_sgd: float = Field(14.0, ge=0)
    avoidable_waste_share: float = Field(0.30, ge=0, le=1)
    recoverable_overstaff_share: float = Field(0.50, ge=0, le=1)


class ClientConfig(BaseModel):
    key: str
    name: str
    inputs_dir: Path
    outputs_dir: Path
    is_sample: bool = False
    weather_station: str = "S24"  # MSS station code; S24 = Changi
    # Lookback for "what happened" figures (waste, staffing gaps, stockouts, item rates).
    review_days: int = 28
    staffing: StaffingParams = StaffingParams()
    staffing_by_outlet: dict[str, dict] = {}
    items: ItemParams = ItemParams()
    items_by_name: dict[str, dict] = {}
    savings: SavingsAssumptions = SavingsAssumptions()

    def staffing_for(self, outlet: str) -> StaffingParams:
        return StaffingParams(**{**self.staffing.model_dump(), **self.staffing_by_outlet.get(outlet, {})})

    def item_params(self, item: str) -> ItemParams:
        return ItemParams(**{**self.items.model_dump(), **self.items_by_name.get(item, {})})

    @property
    def mapping_path(self) -> Path:
        return CLIENTS_DIR / self.key / "mapping.yaml"

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "clients" / self.key / "raw"


def load_client(key: str) -> ClientConfig:
    path = CLIENTS_DIR / key / "client.yaml"
    if not path.exists():
        known = sorted(p.name for p in CLIENTS_DIR.iterdir() if (p / "client.yaml").exists())
        raise SystemExit(f"No client config at {path}. Known clients: {', '.join(known)}")
    raw = yaml.safe_load(path.read_text()) or {}
    raw.setdefault("inputs_dir", f"data/clients/{key}/inputs")
    raw.setdefault("outputs_dir", f"data/clients/{key}/outputs")
    for field in ("inputs_dir", "outputs_dir"):
        raw[field] = ROOT / raw[field]
    return ClientConfig(key=key, **raw)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
