"""New Project Example
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. Every ML model project
should have a definition file like this one.
"""
from typing import Any
from layer import Featureset, Dataset, Train
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def train_model(train: Train,
                application: Dataset("application_train"),
                bureau: Dataset("bureau"),
                installments: Dataset("installments_payments"),
                previous_application: Dataset("previous_application"),
                af: Featureset("application_features"),
                pf: Featureset("previous_application_features"),
                bureau_features: Featureset("bureau_features"),
                ) -> Any:
    """Model train function
    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.
    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml
    Returns:
       model: Trained model object
    """

    application_df = application.to_pandas()

    previous_application_df = previous_application.to_pandas()
    # Datasets
    installments_df = installments.to_pandas()
    installments_df = installments_df[['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
                                       'AMT_INSTALMENT', 'AMT_PAYMENT']]
    bureau = bureau.to_pandas()

    # Featuresets
    application_features_df = af.to_pandas()
    previous_application_features_df = pf.to_pandas()
    bureau_features = bureau_features.to_pandas()

    # Merge feature sets to the dataset
    application_data = application_df.merge(application_features_df, on='INDEX')
    application_data = application_data[['TARGET', 'SK_ID_CURR', 'ANNUITY_INCOME_RATIO', 'CREDIT_INCOME_RATIO',
                                         'CREDIT_TERM', 'DAYS_EMPLOYED_RATIO', 'GOODS_PRICE_LOAN_DIFFERENCE',
                                          'REGION_RATING_CLIENT_W_CITY', 'OWN_CAR_AGE', 'DAYS_BIRTH',
                                         'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY',
                                         'LIVE_CITY_NOT_WORK_CITY', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                                         'FLAG_DOCUMENT_3']]

    bureau_data = bureau.merge(bureau_features, on='INDEX')
    selected_bureau_sample = bureau_data[['SK_ID_CURR', 'CREDIT_LIMIT_ABOVE_ZERO', 'HAS_DEBT',
                                          'AMT_CREDIT_SUM_OVERDUE']]

    previous_application_data = previous_application_df.merge(previous_application_features_df, on='INDEX')
    p_application_df = previous_application_data[['SK_ID_PREV', 'SK_ID_CURR', 'APPLIED_AWARDED_AMOUNT_DIFF',
                                                         'GOODS_PRICE_APPLIED_DIFF']]

    # Merge all of them
    dff = installments_df.merge(selected_bureau_sample, on=['SK_ID_CURR']).merge(application_data,
                                                                           on='SK_ID_CURR').merge(
        p_application_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    # Drop all null rows
    dff = dff.dropna()
    # Obtain the X variable
    X = dff.drop(["SK_ID_PREV", "SK_ID_CURR"], axis=1)
    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the model and calculate
    # the drift
    train.register_input(X)
    # Get all categorical columns
    categories = dff.select_dtypes(include=['object']).columns.tolist()
    # Convert the categorical columns into a numerical representation via one hot encoding
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    # https://scikit-learn.org/stable/modules/compose.html#column-transformer
    transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop="first"), categories)],
        remainder='passthrough')
    # Running dimensionality reduction algorithm such as Principal component analysis (PCA) prior to K-Means reduces
    # the effects of the curse of dimensionality. PCA reduces the number of features. This can be done by either
    # removing or combining features.
    pca = PCA(n_components=2, random_state=42)
    df = transformer.fit_transform(X)
    clustering_data = pca.fit_transform(df)
    sc = StandardScaler()
    # Standardize features by removing the mean and scaling to unit variance
    clustering_data = sc.fit_transform(clustering_data)
    # Model Parameters
    n_clusters = 3
    # Log parameters
    train.log_parameters({"n_clusters": n_clusters})

    # Model: Define a KMeans model
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(clustering_data)
    return kmeans
