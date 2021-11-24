## Tutorial 3: How to create another model by re-using existing features 
In this tutorial we will look how we can use the new features created in tutorial 1 to create a clustering model. In the
fourth tutorial we will use the predictions from this model as input to credit scoring model. 

### Install and run
To checkout the complete project after following this tutorial run:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/3/after_tutorial
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
| |____clustering
| | |____requirements.txt
| | |____model.py
| | |____clustering_model.yaml
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
To run:
```yaml
layer start model churn model
```
```yaml
(layer-env) derrickmwiti@Derricks-MacBook-Pro after_tutorial % layer start model clustering_model  
Layer 0.8.14 using https://beta.layer.co
📁 Loading the project under /Users/derrickmwiti/PycharmProjects/Layer-videos/credit-score/tutorials/3/after_tutorial
🔎 Found 4 datasets, 3 featuresets and 1 model
📔 Session logs at /Users/derrickmwiti/.layer/logs/20211124T133450-session-66472a72-7304-4ebd-a02d-79b4f265b2bb.log
💾 Starting at 2021-11-24 13:34:53...
🔵 Pipeline run id: 31628564-90fd-4f82-a294-0be63c354346
✅ 2021-11-24 13:34:53 | dataset     application_train              ━━━━━━━━━━━━━━━━━━━━━━ DONE      [399ms]                                       
✅ 2021-11-24 13:34:53 | dataset     previous_application           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [783ms]                                       
✅ 2021-11-24 13:34:53 | dataset     installments_payments          ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1165ms]                                      
✅ 2021-11-24 13:34:53 | dataset     bureau                         ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1539ms]                                      
⠏  2021-11-24 13:36:44 | featureset  previous_application_features  ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠏  2021-11-24 13:36:44 | featureset  bureau_features                ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠏  2021-11-24 13:36:44 | featureset  application_features           ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
✅ 2021-11-24 13:35:21 | model       clustering_model               ━━━━━━━━━━━━━━━━━━━━━━ DONE      [82903ms]                                     
                                     https://beta.layer.co/models/692d4296-3d2a-4518-bbda-55294a25597b/trains/c0571ce2-37e9-4b8c-a627-3a26a25d5993 
LAYER RUN SUCCEEDED in 110382ms
```
### Step 1: Update the model file to use the new featureset
First, we need to inject the features and datasets to the model training function:
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

    pass
```
### Step 2: Use the new features to train the model
Let's now use these features to create a clustering model. We use a K-Means clustering algorithm that will later be used
group the dataset into several clusters. 
```python
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
    # Obtain the X and y variables
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

```

### Step 3: Update the model YAML file
Configure the model YAML file to reflect that we are training a new model: 
```yaml
|____models
| |____credit_score
| | |____requirements.txt
| | |____model.py
| | |____clustering_model.yaml
```
```yaml
# New Project Example
#
# Any directory includes an `model.yml` will be treated as a ml model project.
# In this `yaml` file, we will define the attributes of our model.

apiVersion: 1
type: model
# Name and description of our model
name: "clustering_model"
description: "clustering model"

training:
  name: clustering_model_training
  description: "My Model Training"

  # The source model definition file with a `train_model` method
  entrypoint: model.py

  # File includes the required python libraries with their correct versions
  environment: requirements.txt
    # The software and hardware environment needed for this training,
    # as defined in https://docs.beta.layer.co/docs/reference/fabrics
  fabric: "f-medium"
```

### Step 4: Train the model
You can now train this model using the `layer start` command. 
```yaml
layer start model clustering_model 
```

In the next tutorial, we will this model to predict the cluster for each loan and use the to train a credit scoring algorithm. 
See you there. 