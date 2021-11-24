# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(bureau: Dataset("bureau")) -> Any:
    df = bureau.to_pandas()
    df['HAS_OVERDUE_DEBT'] = df['AMT_CREDIT_SUM_OVERDUE'].apply(over_due_debt)
    data = df[['INDEX', 'HAS_OVERDUE_DEBT']]

    return data


def over_due_debt(debt_amount):
    if debt_amount > 0:
        debt_overdue = 1
    else:
        debt_overdue = 0
    return debt_overdue
