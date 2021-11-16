# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(layer_dataset: Dataset("application_train")) -> Any:
    df = layer_dataset.to_pandas()
    df['DAYS_EMPLOYED_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    data = df[['SK_ID_CURR', 'DAYS_EMPLOYED_RATIO']]

    return data
