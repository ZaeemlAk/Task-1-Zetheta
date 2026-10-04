import pandas as pd

from src.data_quality import describe_dataset


def test_describe_dataset_basic_summary():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "gender": ["F", "M", "F"],
        "decision": [1, 0, 1],
        "income": [1000.0, None, 2000.0],
    })

    result = describe_dataset(df)
    assert result["n_records"] == 3
    assert result["n_columns"] == 4
    assert result["duplicate_rows"] == 0
    assert "age" in result["dtypes"]
    assert "gender" in result["dtypes"]
