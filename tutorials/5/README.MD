### Tutorial 5: How to optimize the model by implementing hyper-parameter tuning  
In this tutorial you will learn how to perform [hyperparameter tuning](https://docs.beta.layer.co/docs/models/hyperparametertuning) 
on the credit score model. Layer allows you to tune your model using various strategies such as [Bayesian](https://scikit-optimize.github.io/stable/), 
[Random](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), 
[Grid](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and Manual. 

### Clone project starter
Clone the repo below to follow along with the tutorial:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/5/before_tutorial
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