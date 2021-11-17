# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset
from typing import Any


def build_feature(application_dataset: Dataset("application_train")) -> Any:
    df = application_dataset.to_pandas()
    df['GOODS_PRICE_LOAN_DIFFERENCE'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    data = df[['SK_ID_CURR', 'GOODS_PRICE_LOAN_DIFFERENCE']]

    return data
