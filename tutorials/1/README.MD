## Tutorial 1: How to add new datasets and feature sets to an existing project
On this page, you will learn how to add a new dataset and feature set to an existing Layer project. 

### Install and run
To checkout the complete project after following this tutorial run:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/1/after_tutorial
```
Folder structure:
````yaml
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

````
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
### Step 1: Adding a new dataset
You can add a new dataset to a Layer project by defining it in a `.yaml` file. In this case, we are adding the `bureau` 
dataset to this project. Create a new folder under the data folder and create a `.yaml` file with a preferred name. The
`yaml` file will contain the following information:

````
# For more information on Dataset Configuration: https://docs.beta.layer.co/docs/datacatalog/datasets

apiVersion: 1
# The type of entity
type: dataset
# Unique name of this dataset which will be used in this project to refer to this dataset
name: "bureau"
# The table name and data source integration where the table can be found. 
materialization:
    target: layer-public-datasets
    table_name: "bureau"

````
### Step 2: Create new features
The next step is to use this new dataset to create a new set of features. Under the features folder create a new folder
that will house these features. We'll create three new features has shown below: 
````yaml
| |____bureau_features
| | |____credit_limit.py
| | |____requirements.txt
| | |____has_overdue_debt.py
| | |____has_debt.py
| | |____bureau.yaml
````
Let's take a look at the source code for each individual feature:
#### Credit limit - credit_limit.py
```python
from layer import Dataset
from typing import Any


def build_feature(bureau: Dataset("bureau")) -> Any:
    df = bureau.to_pandas()
    df['CREDIT_LIMIT_ABOVE_ZERO'] = df['AMT_CREDIT_SUM_LIMIT'].apply(credit_sum_limit)
    data = df[['INDEX', 'CREDIT_LIMIT_ABOVE_ZERO']]

    return data


def credit_sum_limit(credit_limit):
    if credit_limit > 0:
        limit_above_zero = 1
    else:
        limit_above_zero = 0
    return limit_above_zero

```
#### Has debt - has_debt.py
```python
from layer import Dataset
from typing import Any


def build_feature(bureau: Dataset("bureau")) -> Any:
    df = bureau.to_pandas()
    df["HAS_DEBT"] = df['AMT_CREDIT_SUM_DEBT'].apply(has_debt)
    data = df[['INDEX', 'HAS_DEBT']]

    return data


def has_debt(debt_amount):
    if debt_amount > 0:
        has_debt = 1
    else:
        has_debt = 0
    return has_debt
```
#### Has overdue debt - has_overdue_debt.py
```python
from layer import Dataset
from typing import Any


def build_feature(bureau: Dataset("bureau")) -> Any:
    df = bureau.to_pandas()
    df['HAS_OVERDUE_DEBT'] = df['AMT_CREDIT_SUM_OVERDUE'].apply(over_due_debt)
    data = df[['INDEX', 'HAS_OVERDUE_DEBT']]

    return data


def over_due_debt(debt_amount):
    if debt_amount > 0:
        debt_overdue = 1
    else:
        debt_overdue = 0
    return debt_overdue
```
### Step 3: Add the feature definition in the YAML file
Next add these features to a YAML file. This file contains the file to the files declaring the features as well as their
description. Create a `bureau.yaml` in the same folder and add the following information: 
```yaml
# For more information on Featureset Configuration: https://docs.beta.layer.co/docs/datacatalog/featuresets

apiVersion: 1

type: featureset

name: "bureau_features"
description: "Previous balance features"
fabric: "f-medium"

features:
    - name: credit_limit_above_zero
      description: "Is the credit limit above zero?"
      source: credit_limit.py
      environment: requirements.txt
    - name: has_debt
      description: "If the customer has debt"
      source: has_debt.py
      environment: requirements.txt
    - name: over_due_debt
      description: "Customer has overdue debt"
      source: has_overdue_debt.py
      environment: requirements.txt

materialization:
    target: layer-public-datasets
```
### Step 4: Build the new features
The defined features can be built using the `layer start featureset` command. 
```yaml
layer start featureset bureau_features
```

Checkout the `after_tutorial` folder for the complete model training code. 

Well done. You now know how to add new data and use it to create a new featureset in Layer. 