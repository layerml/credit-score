from layer import Dataset


def build_feature(layer_dataset: Dataset("layer_dataset_name_for_your_table")) -> Any:
    df = layer_dataset.to_pandas()

    feature_data = df[["column1", "column2"]]

    """
    Your python code goes here
    """

    return feature_data
