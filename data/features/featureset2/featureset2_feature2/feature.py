# Ensure that you have the same ID column across all of your python features.
# Layer joins your singular features using that ID column.

from layer import Dataset


def build_feature(layer_dataset: Dataset("layer_dataset_name_for_your_table")) -> Any:
    df = layer_dataset.to_pandas()

    feature_data = df[["ID_column", "column_name"]]

    """
    Your python code goes here
    """

    return feature_data
