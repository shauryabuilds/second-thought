import pandas as pd
import pytest

from kaizen.config import ClientConfig


@pytest.fixture
def client(tmp_path) -> ClientConfig:
    return ClientConfig(
        key="test", name="Test Café", inputs_dir=tmp_path / "inputs", outputs_dir=tmp_path / "outputs",
        staffing={"transactions_per_staff_hour": 30, "min_staff": 3, "max_staff": 6},
        shifts=[{"name": "Opening", "start": 6, "end": 12}, {"name": "Mid", "start": 12, "end": 18},
                {"name": "Closing", "start": 18, "end": 24}],
        items_by_name={"Croissant": {"pack_size": 12, "cover_days": 1, "perishable": True, "buffer": 0.0},
                       "Beans": {"pack_size": 1, "cover_days": 7, "buffer": 0.1},
                       "Milk": {"pack_size": 12, "cover_days": 3, "buffer": 0.0}},
    )


def hourly_sales(days: int = 3, outlets=("A", "B"), hours=range(8, 12), value: int = 10) -> pd.DataFrame:
    ts = [pd.Timestamp("2026-01-05") + pd.Timedelta(days=d, hours=h) for d in range(days) for h in hours]
    return pd.DataFrame([(t, o, value) for t in ts for o in outlets], columns=["timestamp", "outlet", "transactions"])


def calendar(start: str = "2026-01-01", days: int = 30, holidays=()) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame({"date": dates, "is_public_holiday": dates.isin(pd.to_datetime(list(holidays))).astype(int)})
