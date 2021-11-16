# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(layer_dataset: Dataset("previous_application")) -> Any:
    df = layer_dataset.to_pandas()
    df['APPLIED_AWARDED_AMOUNT_DIFF'] = df['AMT_CREDIT'] - df['AMT_APPLICATION']
    data = df[['SK_ID_CURR', 'APPLIED_AWARDED_AMOUNT_DIFF']]

    return data
