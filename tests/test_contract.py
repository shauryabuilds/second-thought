import pandas as pd
import pytest

from kaizen.contract import SPECS, ContractError, load_inputs, validate_frame
from tests.conftest import hourly_sales


def as_csv_strings(df: pd.DataFrame) -> pd.DataFrame:
    """What read_csv(dtype=str) would give us."""
    out = df.copy()
    if "timestamp" in out:
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.astype(str)


def test_valid_sales_pass_and_get_types():
    df = validate_frame(as_csv_strings(hourly_sales()), SPECS["sales"])
    assert str(df["timestamp"].dtype).startswith("datetime64")
    assert df["transactions"].dtype == "int64"
    assert len(df) == 24


@pytest.mark.parametrize("extra", ["customer_name", "phone", "card_number"])
def test_extra_columns_are_rejected(extra):
    df = as_csv_strings(hourly_sales()).assign(**{extra: "x"})
    with pytest.raises(ContractError, match="Unexpected"):
        validate_frame(df, SPECS["sales"])


def test_missing_column_is_rejected():
    with pytest.raises(ContractError, match="Missing"):
        validate_frame(as_csv_strings(hourly_sales()).drop(columns="outlet"), SPECS["sales"])


@pytest.mark.parametrize("bad, message", [
    ({"timestamp": "2026-01-05 08:15:00"}, "on the hour"),
    ({"timestamp": "2026-01-05T08:00:00+08:00"}, "timezone"),
    ({"transactions": "-3"}, "greater than or equal"),
    ({"transactions": "lots"}, "integer"),
    ({"outlet": ""}, "outlet"),
])
def test_bad_values_are_reported_with_line_numbers(bad, message):
    df = as_csv_strings(hourly_sales())
    for col, value in bad.items():
        df.loc[3, col] = value
    with pytest.raises(ContractError, match=message) as exc:
        validate_frame(df, SPECS["sales"])
    assert "line 5" in str(exc.value)  # row index 3 + header line + 1-based


def test_duplicate_hours_are_rejected():
    df = as_csv_strings(hourly_sales())
    with pytest.raises(ContractError, match="repeat"):
        validate_frame(pd.concat([df, df.head(1)]), SPECS["sales"])


def test_roster_hour_must_be_a_clock_hour():
    roster = pd.DataFrame({"date": ["2026-01-05"], "outlet": ["A"], "hour": ["24"], "scheduled_staff": ["2"]})
    with pytest.raises(ContractError, match="hour"):
        validate_frame(roster, SPECS["roster"])


def test_load_inputs_requires_sales(tmp_path):
    with pytest.raises(ContractError, match="Required file missing"):
        load_inputs(tmp_path)


def test_optional_files_must_use_known_outlets(tmp_path):
    hourly_sales().to_csv(tmp_path / "sales.csv", index=False)
    pd.DataFrame({"date": ["2026-01-05"], "outlet": ["Z"], "waste_sgd": [5.0]}).to_csv(tmp_path / "waste.csv", index=False)
    with pytest.raises(ContractError, match="not found in sales.csv"):
        load_inputs(tmp_path)


def test_optional_files_are_optional(tmp_path):
    hourly_sales().to_csv(tmp_path / "sales.csv", index=False)
    inputs = load_inputs(tmp_path)
    assert inputs.outlets == ["A", "B"]
    assert inputs.roster is None and inputs.stock is None
    assert inputs.history_end == pd.Timestamp("2026-01-07")


def test_generated_sample_matches_the_contract(tmp_path):
    from kaizen.sample import generate

    generate(pd.Timestamp("2026-03-29"), out_dir=tmp_path, allow_network=False, messy_exports=False)
    inputs = load_inputs(tmp_path)
    assert len(inputs.outlets) == 5
    for name in ("items", "stock", "roster", "waste", "menu", "weather"):
        assert getattr(inputs, name) is not None, name
    # Roster runs 7 days past the sales history: that is next week's roster.
    assert inputs.roster["date"].max() == inputs.history_end + pd.Timedelta(days=7)
