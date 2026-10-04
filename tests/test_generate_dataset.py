import pandas as pd

from src.generate_dataset import generate_dataset


def test_generate_dataset_has_expected_columns():
    df = generate_dataset(n=200, output_path=None)

    required = {
        "applicant_id",
        "gender",
        "city_tier",
        "annual_income",
        "credit_score",
        "credit_decision",
        "six_month_default",
    }
    assert required.issubset(df.columns)
    assert len(df) == 200
    assert df["credit_decision"].isin([0, 1]).all()
    assert df["credit_decision"].sum() > 0
