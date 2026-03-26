import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_dataframe(df: pd.DataFrame) -> None:
    required_columns = {"user_id", "skill_id", "correct"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if df["skill_id"].isna().any():
        raise ValueError("Column skill_id contains NaN values.")
    if df["correct"].isna().any():
        raise ValueError("Column correct contains NaN values.")
    if not is_numeric_dtype(df["skill_id"]):
        raise ValueError("Column skill_id must be numeric.")
    if not is_numeric_dtype(df["correct"]):
        raise ValueError("Column correct must be numeric.")

    skill_values = df["skill_id"].astype(float)
    correct_values = df["correct"].astype(float)

    if (skill_values < 0).any():
        raise ValueError("Column skill_id must be non-negative.")
    if not skill_values.apply(lambda value: value.is_integer()).all():
        raise ValueError("Column skill_id must contain integer-like values.")
    if not correct_values.apply(lambda value: value in (0.0, 1.0)).all():
        raise ValueError("Column correct must contain only 0/1 values.")
