## Tutorial 2: How to create a new model version using the newly generated features
On this page, you will learn how you can use the new features created in the last tutorial to create a
new version of the same model. 

### Install and run
Clone the repo below to follow along with the tutorial:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/2/before_tutorial
```
Folder structure:
```yaml
.
|____.layer
| |____project.yaml
|____features
| |____bureau_features
| | |____credit_limit.py
| | |____requirements.txt
| | |____has_overdue_debt.py
| | |____has_debt.py
| | |____bureau.yaml
| |____application_featureset
| | |____application_dataset.yaml
| | |____requirements.txt
| | |____days_employed_ratio.py
| | |____goods_price_diff.py
| | |____credit_income_ratio.py
| | |____annuity_income_ratio.py
| | |____credit_term.py
| |____previous_application_featureset
| | |____requirements.txt
| | |____applied_awarded_amount_diff.py
| | |____previous_app_features.yaml
| | |____goods_price_applied_diff.py
|____models
| |____credit_score
| | |____requirements.txt
| | |____credit_score_model.yaml
| | |____model.py
|____data
| |____installments
| | |____installments_data.yaml
| |____bureau
| | |____bureau.yaml
| |____application_train
| | |____application_data.yaml
| |____previous_application
| | |____previous_application_dataset.yaml

```
To build the project:
```yaml
layer start
```
```yaml
(layer-env) derrickmwiti@Derricks-MacBook-Pro after_tutorial % layer start                                    
Layer 0.8.14 using https://beta.layer.co
📁 Loading the project under /Users/derrickmwiti/PycharmProjects/Layer-videos/credit-score/tutorials/1/after_tutorial
🔎 Found 4 datasets, 3 featuresets and 1 model
📔 Session logs at /Users/derrickmwiti/.layer/logs/20211124T115252-session-491f0d8f-9cbe-4010-ac2f-5b41bc082322.log
💾 Starting at 2021-11-24 11:53:03...
🔵 Pipeline run id: daca5ad5-93e6-4dfc-afa5-7c2ac6803c00
✅ 2021-11-24 11:53:03 | dataset     installments_payments          ━━━━━━━━━━━━━━━━━━━━━━ DONE      [476ms]                                       
✅ 2021-11-24 11:53:03 | dataset     application_train              ━━━━━━━━━━━━━━━━━━━━━━ DONE      [858ms]                                       
✅ 2021-11-24 11:53:03 | dataset     bureau                         ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1272ms]                                      
✅ 2021-11-24 11:53:03 | dataset     previous_application           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1663ms]                                      
✅ 2021-11-24 11:53:29 | featureset  bureau_features                ━━━━━━━━━━━━━━━━━━━━━━ DONE      [76208ms]                                     
                                     https://beta.layer.co/features/3f3a3abe-977e-433c-a360-9df6ef428a15                                           
✅ 2021-11-24 11:53:29 | featureset  application_features           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [101015ms]                                    
                                     https://beta.layer.co/features/6dd8b3fe-7d4b-44db-a511-74c516c0de2d                                           
✅ 2021-11-24 11:53:29 | featureset  previous_application_features  ━━━━━━━━━━━━━━━━━━━━━━ DONE      [76210ms]                                     
                                     https://beta.layer.co/features/d46deeb8-0f0a-45a9-9ee0-d922f1a1163c                                           
✅ 2021-11-24 11:55:10 | model       credit_score_model             ━━━━━━━━━━━━━━━━━━━━━━ DONE      [546404ms]                                    
                                     https://beta.layer.co/models/181c5809-b3b1-4246-a9b2-b882fda417e9/trains/dd168bfc-a70b-4c67-9c23-0bd809d40887 
LAYER RUN SUCCEEDED in 673782ms
```
### Step 1: Update the model file to use the new featureset
Since that the features are ready, update the model training to take advantage of the new features. We can 
do this by injecting the new dataset and features to the model training function:
```python
from typing import Any
from layer import Featureset, Dataset, Train
def train_model(train: Train,
                application: Dataset("application_train"),
                bureau: Dataset("bureau"),
                installments: Dataset("installments_payments"),
                previous_application: Dataset("previous_application"),
                af: Featureset("application_features"),
                pf: Featureset("previous_application_features"),
                bureau_features: Featureset("bureau_features"),
                ) -> Any:

    bureau = bureau.to_pandas()
    bureau_features = bureau_features.to_pandas()
```
### Step 2: Use the new features to train the model
The next step is to use the newly generated features to train a new version of the same model. A new model version will 
created when you train the model. Here's the complete model code:
```python
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
                bureau: Dataset("bureau"),
                installments: Dataset("installments_payments"),
                previous_application: Dataset("previous_application"),
                af: Featureset("application_features"),
                pf: Featureset("previous_application_features"),
                bureau_features: Featureset("bureau_features"),
                ) -> Any:
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
    # Select the relevant columns
    application_data = application_data[['TARGET', 'SK_ID_CURR', 'ANNUITY_INCOME_RATIO', 'CREDIT_INCOME_RATIO',
                                         'CREDIT_TERM', 'DAYS_EMPLOYED_RATIO', 'GOODS_PRICE_LOAN_DIFFERENCE',
                                          'REGION_RATING_CLIENT_W_CITY', 'OWN_CAR_AGE', 'DAYS_BIRTH',
                                         'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY',
                                         'LIVE_CITY_NOT_WORK_CITY', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                                         'FLAG_DOCUMENT_3']]

    # Merge the bureau dataset to the newly created features
    bureau_data = bureau.merge(bureau_features, on='INDEX')
    # Select the relevant columns
    selected_bureau_sample = bureau_data[['SK_ID_CURR', 'CREDIT_LIMIT_ABOVE_ZERO', 'HAS_DEBT',
                                          'AMT_CREDIT_SUM_OVERDUE']]
    # Merge the previous application dataset to the newly created features
    previous_application_data = previous_application_df.merge(previous_application_features_df, on='INDEX')
    p_application_df = previous_application_data[['SK_ID_PREV', 'SK_ID_CURR', 'APPLIED_AWARDED_AMOUNT_DIFF',
                                                         'GOODS_PRICE_APPLIED_DIFF']]

    # Merge all of them
    dff = installments_df.merge(selected_bureau_sample, on=['SK_ID_CURR']).merge(application_data,
                                                                           on='SK_ID_CURR').merge(
        p_application_df, on=['SK_ID_PREV', 'SK_ID_CURR'])
    # Drop all null rows
    dff = dff.dropna()
    # Obtain the X and y variables
    X = dff.drop(["SK_ID_PREV", "SK_ID_CURR", "TARGET"], axis=1)
    y = dff["TARGET"]
    # Split the data into a training and testing set
    random_state = 13
    test_size = 0.3
    # Log parameters, these can be used for comparing different models on the model catalog
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
    model_random_state = 42
    early_stopping = True
    # Log parameters
    train.log_parameters({"learning_rate": learning_rate,
                          "max_depth": max_depth,
                          "test_size": test_size,
                          "min_samples_leaf": min_samples_leaf,
                          "random_state": model_random_state,
                          "early_stopping": early_stopping,
                          "model_random_state": model_random_state})

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
    # Calculate average precision and area under the receiver operating characteristic curve (ROC AUC)
    avg_precision = average_precision_score(y_test, probs, pos_label=1)
    auc = roc_auc_score(y_test, probs)
    train.log_metric("avg_precision", avg_precision)
    train.log_metric("auc", auc)
    return pipeline

```
### Step 3: Train the model 
The final step is to train a new model using the new features. The model is trained using the `layer start model` command.
```yaml
layer start model credit_score_model
```
```yaml
(layer-env) derrickmwiti@Derricks-MacBook-Pro after_tutorial % layer start model credit_score_model
Layer 0.8.14 using https://beta.layer.co
📁 Loading the project under /Users/derrickmwiti/PycharmProjects/Layer-videos/credit-score/tutorials/2/after_tutorial
🔎 Found 4 datasets, 3 featuresets and 1 model
📔 Session logs at /Users/derrickmwiti/.layer/logs/20211124T130440-session-3e7b006d-84e6-490b-8450-3f6dc73c4fd0.log
💾 Starting at 2021-11-24 13:04:44...
🔵 Pipeline run id: 8aa1fc6f-0c0f-4530-813a-ac17e264f72e
✅ 2021-11-24 13:04:44 | dataset     application_train              ━━━━━━━━━━━━━━━━━━━━━━ DONE      [409ms]                                       
✅ 2021-11-24 13:04:44 | dataset     previous_application           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [837ms]                                       
✅ 2021-11-24 13:04:44 | dataset     bureau                         ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1231ms]                                      
✅ 2021-11-24 13:04:44 | dataset     installments_payments          ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1630ms]                                      
⠧  2021-11-24 13:14:14 | featureset  previous_application_features  ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠧  2021-11-24 13:14:14 | featureset  application_features           ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠧  2021-11-24 13:14:14 | featureset  bureau_features                ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
✅ 2021-11-24 13:05:12 | model       credit_score_model             ━━━━━━━━━━━━━━━━━━━━━━ DONE      [541804ms]                                    
                                     https://beta.layer.co/models/181c5809-b3b1-4246-a9b2-b882fda417e9/trains/1819ec47-0689-447d-9a31-3c234605a55e 
LAYER RUN SUCCEEDED in 570194ms
```
You have now seen how to use newly generated features to create a new version of the same model. In the next tutorial, 
we will look at how you can use the same features to create a different type of model. 