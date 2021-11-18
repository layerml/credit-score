# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(application_dataset: Dataset("application_train")) -> Any:
    df = application_dataset.to_pandas()
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    data = df[['INDEX', 'CREDIT_TERM']]

    return data
