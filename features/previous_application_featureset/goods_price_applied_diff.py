# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(previous_application_dataset: Dataset("previous_application")) -> Any:
    df = previous_application_dataset.to_pandas()
    # df = df.sample(1000, random_state=0)
    df['GOODS_PRICE_APPLIED_DIFF'] = df['AMT_GOODS_PRICE'] - df['AMT_APPLICATION']
    data = df[['INDEX', 'GOODS_PRICE_APPLIED_DIFF']]

    return data
