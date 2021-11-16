# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(layer_dataset: Dataset("application_train")) -> Any:
    df = layer_dataset.to_pandas()
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    data = df[['SK_ID_CURR', 'CREDIT_TERM']]

    return data
