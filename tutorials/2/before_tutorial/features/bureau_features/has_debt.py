# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(bureau: Dataset("bureau")) -> Any:
    df = bureau.to_pandas()
    df["HAS_DEBT"] = df['AMT_CREDIT_SUM_DEBT'].apply(has_debt)
    data = df[['INDEX', 'HAS_DEBT']]

    return data


def has_debt(debt_amount):
    if debt_amount > 0:
        has_debt = 1
    else:
        has_debt = 0
    return has_debt
