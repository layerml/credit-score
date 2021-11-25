### Tutorial 5: How to optimize the model by implementing hyper-parameter tuning  
In this tutorial you will learn how to perform [hyperparameter tuning](https://docs.beta.layer.co/docs/models/hyperparametertuning) 
on the credit score model. Layer allows you to tune your model using various strategies such as [Bayesian](https://scikit-optimize.github.io/stable/), 
[Random](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), 
[Grid](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and Manual. 

### Install and run
To checkout the complete project after following this tutorial run:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/5/after_tutorial
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
To run: 
```yaml
(layer-env) derrickmwiti@Derricks-MacBook-Pro after_tutorial % layer start model credit_score_model           
Layer 0.8.14 using https://beta.layer.co
📁 Loading the project under /Users/derrickmwiti/PycharmProjects/Layer-videos/credit-score/tutorials/5/after_tutorial
🔎 Found 4 datasets, 3 featuresets and 1 model
📔 Session logs at /Users/derrickmwiti/.layer/logs/20211124T201614-session-b3aa9d22-5a56-4fab-acef-9dc8a57de43c.log
💾 Starting at 2021-11-24 20:16:17...
🔵 Pipeline run id: 0111d7d1-b034-4fa1-a9a5-7b7bda680970
✅ 2021-11-24 20:16:17 | dataset     previous_application           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [406ms]                                       
✅ 2021-11-24 20:16:17 | dataset     application_train              ━━━━━━━━━━━━━━━━━━━━━━ DONE      [785ms]                                       
✅ 2021-11-24 20:16:17 | dataset     installments_payments          ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1148ms]                                      
✅ 2021-11-24 20:16:17 | dataset     bureau                         ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1521ms]                                      
⠸  2021-11-24 20:49:32 | featureset  bureau_features                ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠸  2021-11-24 20:49:32 | featureset  previous_application_features  ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠸  2021-11-24 20:49:32 | featureset  application_features           ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
✅ 2021-11-24 20:16:52 | model       credit_score_model             ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1960684ms]                                   
                                     https://beta.layer.co/models/181c5809-b3b1-4246-a9b2-b882fda417e9/trains/7d674b54-c62f-4333-92eb-7f316445d77b 
LAYER RUN SUCCEEDED in 1995499ms

```
### Step 1: Add tuning parameters 
The first step is to update the model YAML file with the search parameters. These are added under the `hyperparameters` 
key. In this case, we'll implement the Random search strategy. We maximize the average precision. 

Here is the updated YAML file:
```yaml
apiVersion: 1
type: model
# Name and description of our model
name: "credit_score_model"
description: "Credit score model"

training:
  name: credit_score_model_training
  description: "My Model Training"

  # The source model definition file with a `train_model` method
  entrypoint: model.py
  # https://docs.beta.layer.co/docs/models/hyperparametertuning
  hyperparameters:
    strategy: "Random"
    max_trials: 5
    maximize: "avg_precision"
    tuning:
      learning_rate:
        type: "float"
        min: 0.01
        max: 0.9
      l2_regularization:
        type: "float"
        min: 0.0
        max: 0.9
      max_depth:
        type: "integer"
        values: [ 1, 3, 5,6 ]
      max_iter:
        type: "integer"
        values: [ 100, 300, 500,600 ]
      min_samples_leaf:
        type: "integer"
        values: [ 10, 30, 50,60 ]

  # File includes the required Python libraries with their correct versions
  environment: requirements.txt
    # The software and hardware environment needed for this training,
    # as defined in https://docs.beta.layer.co/docs/reference/fabrics
  fabric: "f-medium"
```
### Step 2: Update model to use the parameters
The next step is to update the model to use the defined parameters. We use the `train.get_parameter` function to 
obtain the parameters defined in the model YAML file. 
```python
learning_rate = train.get_parameter("learning_rate")
l2_regularization = train.get_parameter("l2_regularization")
max_depth = train.get_parameter("max_depth")
max_iter = train.get_parameter("max_iter")
min_samples_leaf = train.get_parameter("min_samples_leaf")
model_random_state = 42
model = HistGradientBoostingClassifier(learning_rate=learning_rate,
                                           l2_regularization=l2_regularization,
                                           max_depth=max_depth,
                                           max_iter=max_iter,
                                           min_samples_leaf=min_samples_leaf,
                                           random_state=model_random_state)
```

You have now seen how to implement hyperparameter tuning for your Layer project. 