## Tutorial 1: How to add new datasets and feature sets to an existing project
On this page, you will learn how to add a new dataset and feature set to an existing Layer project. 

### Clone project starter
Clone the repo below to follow along with the tutorial:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/1/before_tutorial
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

Checkout the `after_tutorial` folder for the complete project code. 

Well done. You now know how to add new data and use it to create a new featureset in Layer. 