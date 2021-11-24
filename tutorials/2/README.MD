## Tutorial 2: How to create a new model version using the newly generated features

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