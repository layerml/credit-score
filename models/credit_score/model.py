"""New Project Example
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. Every ML model project
should have a definition file like this one.
"""
from typing import Any
from layer import Featureset, Dataset, Train
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
# This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).
# https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
from sklearn.ensemble import HistGradientBoostingClassifier

def train_model(train: Train,
                application: Dataset("application_train"),
                pos: Dataset("POS_CASH_balance"),
                installments: Dataset("installments_payments"),
                previous_application: Dataset("previous_application"),
                af: Featureset("application_features"),
                pf: Featureset("previous_application_features"),

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
    columns = ['OWN_CAR_AGE', 'OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'APARTMENTS_AVG',
               'BASEMENTAREA_AVG',
               'YEARS_BEGINEXPLUATATION_AVG', 'COMMONAREA_AVG', 'YEARS_BUILD_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
               'FLOORSMAX_AVG',
               'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
               'APARTMENTS_MODE',
               'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
               'ELEVATORS_MODE', 'ELEVATORS_MODE',
               'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
               'NONLIVINGAREA_MODE',
               'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
               'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
               'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
               'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
               'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
               'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
               'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
               'NONLIVINGAREA_AVG', 'LIVINGAREA_MODE', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MODE']

    application_df = application_df.drop(columns=columns, axis=1)
    previous_application_df = previous_application.to_pandas()
    previous_columns = ['AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
                        'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
                        'DAYS_TERMINATION',
                        'NFLAG_INSURED_ON_APPROVAL']
    # Datasets
    previous_application_df = previous_application_df.drop(columns=previous_columns, axis=1)
    installments_df = installments.to_pandas()
    pos_df = pos.to_pandas()
    # Featuresets
    application_features_df = af.to_pandas()
    previous_application_features_df = pf.to_pandas()
    # Merge featuresets to the dataset
    application_data = application_df.merge(application_features_df, on='SK_ID_CURR')
    previous_application_data = application_df.merge(previous_application_features_df, on='SK_ID_CURR')
    # Merge all of them
    dff = installments_df.merge(previous_application_data, on=['SK_ID_PREV', 'SK_ID_CURR']).merge(
                           pos_df, on=['SK_ID_PREV', 'SK_ID_CURR']).merge(application_data, on='SK_ID_CURR')
    # Obtain the X and y variables
    X = dff.drop(["SK_ID_PREV", "SK_ID_CURR", "TARGET"], axis=1)
    y = dff["TARGET"]
    # Split the data into a training and testing set
    random_state = 13
    test_size = 0.3
    # Log parameters, these can be used for comparing different models on the model catalog
    train.log_parameter("test_size", test_size)
    train.log_parameter("random_state", random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)
    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the model and calculate
    # the drift
    train.register_input(X_train)
    train.register_output(dff['TARGET'])
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
    # Model Parameters
    learning_rate = 0.01
    max_depth = 6
    min_samples_leaf = 10
    random_state = 42
    early_stopping = True
    # Log model parameters
    train.log_parameter("learning_rate", learning_rate)
    train.log_parameter("max_depth", max_depth)
    train.log_parameter("min_samples_leaf", min_samples_leaf)
    train.log_parameter("random_state", random_state)
    train.log_parameter("early_stopping", early_stopping)
    # Model: Define a HistGradient Boosting Classifier
    model = HistGradientBoostingClassifier(learning_rate=learning_rate,
                                           max_depth=max_depth,
                                           min_samples_leaf=min_samples_leaf,
                                           early_stopping=early_stopping,
                                           random_state=random_state)

    # Fit the pipeline
    pipeline = Pipeline(steps=[('transformer', transformer), ('model', model)])
    pipeline.fit(X_train, y_train)
    # Predict probabilities of target
    probs = pipeline.predict_proba(X_test)[:, 1]
    # Calculate average precision and area under the receiver operating characteric curve (ROC AUC)
    avg_precision = average_precision_score(y_test, probs, pos_label=1)
    auc = roc_auc_score(y_test, probs)
    train.log_metric("avg_precision", avg_precision)
    train.log_metric("auc", auc)
    return pipeline



