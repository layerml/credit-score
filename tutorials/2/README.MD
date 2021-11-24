## Tutorial 2: How to create a new model version using the newly generated features


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
### Step 5: Update the model file to use the new featureset
Now that the features are ready, you can also update your model training to take advantage of the new features. We can 
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