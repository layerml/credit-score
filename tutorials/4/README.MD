### Tutorial 4: How to use the results of the new model to create a new feature and use it to train another model 
In the last tutorial we used the newly created features to create a clustering model. In this tutorial, we will use that
model to create a new feature known as `cluster`. We'll then add this feature to the existing training data and train a 
new credit scoring model. 
### Install and run
To checkout the complete project after following this tutorial run:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/4/after_tutorial
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
(layer-env) derrickmwiti@Derricks-MacBook-Pro after_tutorial % layer start
Layer 0.8.14 using https://beta.layer.co
📁 Loading the project under /Users/derrickmwiti/PycharmProjects/Layer-videos/credit-score/tutorials/4/after_tutorial
🔎 Found 4 datasets, 3 featuresets and 1 model
📔 Session logs at /Users/derrickmwiti/.layer/logs/20211124T141324-session-6463a746-b8ac-4975-bef0-bc8d723eb389.log
💾 Starting at 2021-11-24 14:13:27...
🔵 Pipeline run id: ae142f41-fbce-4e15-b728-266739f0cb2a
✅ 2021-11-24 14:13:27 | dataset     previous_application           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [425ms]                                       
✅ 2021-11-24 14:13:27 | dataset     bureau                         ━━━━━━━━━━━━━━━━━━━━━━ DONE      [958ms]                                       
✅ 2021-11-24 14:13:27 | dataset     installments_payments          ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1503ms]                                      
✅ 2021-11-24 14:13:27 | dataset     application_train              ━━━━━━━━━━━━━━━━━━━━━━ DONE      [2056ms]                                      
✅ 2021-11-24 14:13:59 | featureset  previous_application_features  ━━━━━━━━━━━━━━━━━━━━━━ DONE      [75554ms]                                     
                                     https://beta.layer.co/features/d46deeb8-0f0a-45a9-9ee0-d922f1a1163c                                           
✅ 2021-11-24 14:13:59 | featureset  bureau_features                ━━━━━━━━━━━━━━━━━━━━━━ DONE      [75580ms]                                     
                                     https://beta.layer.co/features/3f3a3abe-977e-433c-a360-9df6ef428a15                                           
✅ 2021-11-24 14:13:59 | featureset  application_features           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [92108ms]                                     
                                     https://beta.layer.co/features/6dd8b3fe-7d4b-44db-a511-74c516c0de2d                                           
✅ 2021-11-24 14:15:31 | model       credit_score_model             ━━━━━━━━━━━━━━━━━━━━━━ DONE      [547645ms]                                    
                                     https://beta.layer.co/models/181c5809-b3b1-4246-a9b2-b882fda417e9/trains/81b52cbd-b2d9-49ea-a1ad-431bfdde26df
```
### Step 1: Data transformation
The first step is to transform the data that will use to run prediction to be similar to the data used to train the 
clustering model. The transformation is similar to the one done in the previous tutorial. 

```python
clustering_data = dff.drop(["SK_ID_PREV", "SK_ID_CURR"], axis=1)
# Get all categorical columns
categories = dff.select_dtypes(include=['object']).columns.tolist()

transformer = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop="first"), categories)],
    remainder='passthrough')
pca = PCA(n_components=2, random_state=42)
clustering_data = transformer.fit_transform(clustering_data)
clustering_data = pca.fit_transform(clustering_data)
sc = StandardScaler()
# Standardize features by removing the mean and scaling to unit variance
clustering_data = sc.fit_transform(clustering_data)
```

### Step 2: Fetch the trained model
The next step is to fetch the trained clustering model:
```python
clustering_model = layer.get_model("clustering_model")
clustering_model_train = clustering_model.get_train()
```

### Step 3: Make predictions
The next step is to use the trained model to predict clusters: 
```python
predictions = clustering_model_train.predict(clustering_data)
```

### Step 4: Append the clusters to the training data
The final step is to create a new column and append the predicted clusters: 
```python
dff['cluster'] = predictions
```
### Step 4: Run the model
Next, train the new clusterig model:
```yaml
layer start model credit_score_model
```

```yaml
Layer 0.8.14 using https://beta.layer.co
📁 Loading the project under /Users/derrickmwiti/PycharmProjects/Layer-videos/credit-score/tutorials/4/after_tutorial
🔎 Found 4 datasets, 3 featuresets and 1 model
📔 Session logs at /Users/derrickmwiti/.layer/logs/20211124T143149-session-184d2b40-3811-4c49-a43b-2e928b97c0e5.log
💾 Starting at 2021-11-24 14:31:52...
🔵 Pipeline run id: 0312a72f-ef8d-4afe-a240-e3bdb043c177
✅ 2021-11-24 14:31:52 | dataset     installments_payments          ━━━━━━━━━━━━━━━━━━━━━━ DONE      [405ms]                                       
✅ 2021-11-24 14:31:52 | dataset     application_train              ━━━━━━━━━━━━━━━━━━━━━━ DONE      [779ms]                                       
✅ 2021-11-24 14:31:52 | dataset     bureau                         ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1150ms]                                      
✅ 2021-11-24 14:31:52 | dataset     previous_application           ━━━━━━━━━━━━━━━━━━━━━━ DONE      [1714ms]                                      
⠼  2021-11-24 14:41:34 | featureset  application_features           ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠼  2021-11-24 14:41:34 | featureset  previous_application_features  ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
⠼  2021-11-24 14:41:34 | featureset  bureau_features                ━━━━━━━━━━━━━━━━━━━━━━ PENDING   [0ms]                                         
✅ 2021-11-24 14:32:16 | model       credit_score_model             ━━━━━━━━━━━━━━━━━━━━━━ DONE      [558213ms]                                    
                                     https://beta.layer.co/models/181c5809-b3b1-4246-a9b2-b882fda417e9/trains/c0d0a77b-360f-4d87-adda-ca904b78c9ef 
LAYER RUN SUCCEEDED in 581994ms
```