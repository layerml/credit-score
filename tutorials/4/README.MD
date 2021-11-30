### Tutorial 4: How to use the results of the new model to create a new feature and use it to train another model 
In the last tutorial we used the newly created features to create a clustering model. In this tutorial, we will use that
model to create a new feature known as `cluster`. We'll then add this feature to the existing training data and train a 
new credit scoring model. 
### Clone project starter
Clone the repo below to follow along with the tutorial:
```yaml
layer clone https://github.com/layerml/credit-score.git
cd credit-score/tutorials/4/before_tutorial
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
(layer-env) derrickmwiti@Derricks-MacBook-Pro after_tutorial % layer start model credit_score_model
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