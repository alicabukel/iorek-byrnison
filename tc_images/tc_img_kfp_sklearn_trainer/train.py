import os
import pickle
import subprocess
import sys

import fire
import hypertune
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]
MODEL_FILENAME = "model.pkl"


def train_evaluate(
    training_dataset_path, validation_dataset_path, alpha, max_iter, hptune
):
    col_map = [('customerID', 'key'),
     ('gender', 'cat'),
     ('SeniorCitizen', 'cat'),
     ('Partner', 'cat'),
     ('Dependents', 'cat'),
     ('tenure', 'num'),
     ('PhoneService', 'cat'),
     ('MultipleLines', 'cat'),
     ('InternetService', 'cat'),
     ('OnlineSecurity', 'cat'),
     ('OnlineBackup', 'cat'),
     ('DeviceProtection', 'cat'),
     ('TechSupport', 'cat'),
     ('StreamingTV', 'cat'),
     ('StreamingMovies', 'cat'),     
     ('Contract', 'cat'),
     ('PaperlessBilling', 'cat'),
     ('PaymentMethod', 'cat'),
     ('MonthlyCharges', 'num'),
     ('TotalCharges', 'num'),
     ('Churn', 'label')] 
     
    df_train = pd.read_csv(training_dataset_path, na_values=' ', header=None, names=[x[0] for x in col_map])
    df_validation = pd.read_csv(validation_dataset_path, na_values=' ', header=None, names=[x[0] for x in col_map])

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_train['TotalCharges'] = simple_imputer.fit_transform(df_train[['TotalCharges']])
    df_validation['TotalCharges'] = simple_imputer.transform(df_validation[['TotalCharges']])

    
    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    numeric_features = [x[0] for x in col_map if x[1] == 'num']

    categorical_features = [x[0] for x in col_map if x[1] == 'cat']

    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", SGDClassifier(loss="log")),
        ]
    )

    num_features_type_map = {feature: "float64" for feature in numeric_features}
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)

    print(f"Starting training: alpha={alpha}, max_iter={max_iter}")
    X_train = df_train.drop("Churn", axis=1)
    y_train = df_train["Churn"]

    pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipeline.fit(X_train, y_train)

    if hptune:
        X_validation = df_validation.drop("Churn", axis=1)
        y_validation = df_validation["Churn"]
        accuracy = pipeline.score(X_validation, y_validation)
        print(f"Model accuracy: {accuracy}")
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="accuracy", metric_value=accuracy
        )

    if not hptune:
        with open(MODEL_FILENAME, "wb") as model_file:
            pickle.dump(pipeline, model_file)
        subprocess.check_call(
            ["gsutil", "cp", MODEL_FILENAME, AIP_MODEL_DIR], stderr=sys.stdout
        )
        print(f"Saved model in: {AIP_MODEL_DIR}")


if __name__ == "__main__":
    fire.Fire(train_evaluate)
