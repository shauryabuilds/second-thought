import pandas as pd
import pytest

from kaizen.ingest import SalesMapping, drop_partial_last_day, ingest_sales


def write(raw_dir, name, rows, columns):
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(raw_dir / name, index=False)


def test_15_minute_buckets_are_summed_to_hours_in_local_time(tmp_path):
    rows = [
        ("2026-01-05T08:00:00+08:00", "STORE 1", 3, "12.50"),
        ("2026-01-05T08:15:00+08:00", "STORE 1", 4, "20.00"),
        ("2026-01-05T08:45:00+08:00", "STORE 1", 1, "4.00"),
        ("2026-01-05T01:30:00+00:00", "STORE 1", 2, "9.00"),  # 09:30 Singapore time
    ]
    write(tmp_path, "export.csv", rows, ["Interval", "Branch", "Bills", "Gross Sales"])
    m = SalesMapping(file="export.csv", layout="buckets", timestamp="Interval", outlet="Branch", count="Bills")
    sales = ingest_sales(m, {"STORE 1": "Orchard"}, tmp_path)
    assert sales.to_dict("records") == [
        {"timestamp": pd.Timestamp("2026-01-05 08:00"), "outlet": "Orchard", "transactions": 8},
        {"timestamp": pd.Timestamp("2026-01-05 09:00"), "outlet": "Orchard", "transactions": 2},
    ]


def test_transaction_rows_count_receipts_once_and_skip_voids(tmp_path):
    rows = [
        ("R1", "05/01/2026", "08:05:00", "S1", "COMPLETED"),
        ("R1", "05/01/2026", "08:05:00", "S1", "COMPLETED"),  # second line on the same receipt
        ("R2", "05/01/2026", "08:40:10", "S1", "COMPLETED"),
        ("R3", "05/01/2026", "08:50:00", "S1", "VOID"),
        ("R4", "05/01/2026", "13:00:00", "S1", "COMPLETED"),
    ]
    write(tmp_path, "txns.csv", rows, ["Receipt", "Date", "Time", "Store", "Status"])
    m = SalesMapping(file="txns.csv", layout="transactions", date="Date", time="Time",
                     timestamp_format="%d/%m/%Y %H:%M:%S", outlet="Store", transaction_id="Receipt",
                     exclude={"Status": ["VOID"]})
    sales = ingest_sales(m, {"S1": "Tampines"}, tmp_path)
    assert sales["transactions"].tolist() == [2, 1]
    assert sales["timestamp"].dt.hour.tolist() == [8, 13]


def test_columns_outside_the_mapping_are_never_read(tmp_path, monkeypatch):
    rows = [("2026-01-05 08:00:00", "S1", 5, "XXXX", "0000 0000")]  # placeholder values, not real data
    write(tmp_path, "export.csv", rows, ["When", "Store", "Count", "Customer Name", "Phone"])
    seen = {}
    real_read_csv = pd.read_csv

    def spy(*args, **kwargs):
        df = real_read_csv(*args, **kwargs)
        if len(df):  # reading only the header row to check column names is fine
            seen.setdefault("cols", set()).update(df.columns)
        return df

    monkeypatch.setattr(pd, "read_csv", spy)
    m = SalesMapping(file="export.csv", layout="buckets", timestamp="When", outlet="Store", count="Count")
    ingest_sales(m, {}, tmp_path)
    assert "Customer Name" not in seen["cols"] and "Phone" not in seen["cols"]


def test_unknown_outlets_stop_the_ingest(tmp_path):
    write(tmp_path, "e.csv", [("2026-01-05 08:00:00", "NEW STORE", 5)], ["When", "Store", "Count"])
    m = SalesMapping(file="e.csv", layout="buckets", timestamp="When", outlet="Store", count="Count")
    with pytest.raises(SystemExit, match="NEW STORE"):
        ingest_sales(m, {"S1": "Orchard"}, tmp_path)


def test_mapping_needs_a_time_source():
    with pytest.raises(ValueError):
        SalesMapping(file="x.csv", layout="buckets", outlet="Store", count="Count")


def test_partial_last_day_is_dropped():
    full = [pd.Timestamp("2026-01-01") + pd.Timedelta(days=d, hours=h) for d in range(8) for h in range(8, 22)]
    partial = [pd.Timestamp("2026-01-09") + pd.Timedelta(hours=h) for h in range(8, 12)]
    sales = pd.DataFrame({"timestamp": full + partial, "outlet": "A", "transactions": 5})
    out = drop_partial_last_day(sales)
    assert out["timestamp"].max().normalize() == pd.Timestamp("2026-01-08")
