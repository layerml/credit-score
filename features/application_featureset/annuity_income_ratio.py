# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(application_dataset: Dataset("application_train")) -> Any:
    df = application_dataset.to_pandas()
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    data = df[['SK_ID_CURR', 'ANNUITY_INCOME_RATIO']]

    return data
