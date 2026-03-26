import pandas as pd
import pytest

from knowledge_tracing.data.validation import validate_dataframe


def test_validate_dataframe_accepts_valid_input():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "skill_id": [0, 1, 2],
            "correct": [0, 1, 0],
        }
    )

    validate_dataframe(df)


def test_validate_dataframe_rejects_non_binary_correct():
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "skill_id": [0, 1],
            "correct": [0, 2],
        }
    )

    with pytest.raises(ValueError, match="0/1"):
        validate_dataframe(df)


def test_validate_dataframe_rejects_non_integer_skill_id():
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "skill_id": [0.0, 1.5],
            "correct": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="integer-like"):
        validate_dataframe(df)
