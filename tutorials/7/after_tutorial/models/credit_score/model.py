"""New Project Example
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. Every ML model project
should have a definition file like this one.
"""
from typing import Any
from layer import Dataset, Train
import layer
# This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).
# https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

def train_model(train: Train,
                application: Dataset("application_train"),
                bureau: Dataset("bureau"),
                installments: Dataset("installments_payments"),
                previous_application: Dataset("previous_application")
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

    application_df = application.to_spark()

    previous_application_df = previous_application.to_spark()
    # Datasets
    installments_df = installments.to_spark()
    installments_df = installments_df[['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
                                       'AMT_INSTALMENT', 'AMT_PAYMENT']]
    bureau = bureau.to_spark()

    # Featuresets
    application_features_df = layer.get_featureset("application_features").to_spark()
    previous_application_features_df = layer.get_featureset("previous_application_features").to_spark()
    bureau_features = layer.get_featureset("bureau_features").to_spark()

    # Merge feature sets to the dataset
    application_data = application_df.join(application_features_df, ['INDEX'])
    application_data = application_data[['TARGET', 'SK_ID_CURR', 'ANNUITY_INCOME_RATIO', 'CREDIT_INCOME_RATIO',
                                         'CREDIT_TERM', 'DAYS_EMPLOYED_RATIO', 'GOODS_PRICE_LOAN_DIFFERENCE',
                                          'REGION_RATING_CLIENT_W_CITY', 'OWN_CAR_AGE', 'DAYS_BIRTH',
                                         'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY',
                                         'LIVE_CITY_NOT_WORK_CITY', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                                         'FLAG_DOCUMENT_3']]

    bureau_data = bureau.join(bureau_features, ['INDEX'])
    selected_bureau_sample = bureau_data[['SK_ID_CURR', 'CREDIT_LIMIT_ABOVE_ZERO', 'HAS_DEBT',
                                          'AMT_CREDIT_SUM_OVERDUE']]

    previous_application_data = previous_application_df.join(previous_application_features_df, ['INDEX'])
    p_application_df = previous_application_data[['SK_ID_PREV', 'SK_ID_CURR', 'APPLIED_AWARDED_AMOUNT_DIFF',
                                                         'GOODS_PRICE_APPLIED_DIFF']]

    # Join all of them
    dff = installments_df.join(selected_bureau_sample, ['SK_ID_CURR']).join(application_data, ['SK_ID_CURR']).join(
        p_application_df, ['SK_ID_PREV', 'SK_ID_CURR'])
    # Drop all null rows
    dff = dff.dropna()
    # Obtain the X and y variables
    feat_cols = application_data.columns
    feat_cols.remove('TARGET')
    label_col = 'TARGET'
    vec_assember = VectorAssembler(inputCols=feat_cols, outputCol='features')
    final_data = vec_assember.transform(dff)
    # Split the data into a training and testing set
    training_size = 0.8
    random_state = 0
    test_size = 0.3
    # Log parameters, these can be used for comparing different models on the model catalog
    training, testing = final_data.randomSplit([training_size, test_size], seed=random_state)
    # Model parameters
    labelCol = label_col
    featuresCol = 'features'
    maxDepth = 5
    maxBins = 32
    seed = 0
    maxIter = 20
    # log model parameters
    train.log_parameters({
        "labelCol": labelCol,
        "featuresCol": featuresCol,
        "maxDepth": maxDepth,
        "maxBins": maxBins,
        "seed": seed,
        "maxIter": maxIter
    })
    lr = GBTClassifier(labelCol=labelCol, featuresCol=featuresCol,
                       maxDepth=maxDepth, maxBins=maxBins, seed=seed, maxIter=maxIter)
    credit_model = lr.fit(training)
    predictions = credit_model.transform(testing)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)

    train.log_metric("areaUnderROC", evaluator.evaluate(predictions))

    return credit_model
