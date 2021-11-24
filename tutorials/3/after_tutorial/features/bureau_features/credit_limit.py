# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(bureau: Dataset("bureau")) -> Any:
    df = bureau.to_pandas()
    df['CREDIT_LIMIT_ABOVE_ZERO'] = df['AMT_CREDIT_SUM_LIMIT'].apply(credit_sum_limit)
    data = df[['INDEX', 'CREDIT_LIMIT_ABOVE_ZERO']]

    return data


def credit_sum_limit(credit_limit):
    if credit_limit > 0:
        limit_above_zero = 1
    else:
        limit_above_zero = 0
    return limit_above_zero
