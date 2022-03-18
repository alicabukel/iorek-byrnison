
import pickle
import subprocess
import sys
import datetime
import os

import fire
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]

def train_evaluate(training_dataset_path, validation_dataset_path):
    
    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)
    df_train = pd.concat([df_train, df_validation])
    
    df_train['Churn'] = df_train['Churn'].map({'Yes': 1, 'No': 2})
    df_validation['Churn'] = df_validation['Churn'].map({'Yes': 1, 'No': 2})
    
    numeric_features = [
        'TotalCharges', 'MonthlyCharges','tenure'
    ]
    
    categorical_features = ['SeniorCitizen', 'Contract', 'TechSupport', 'OnlineSecurity',
                           'InternetService', 'PaperlessBilling', 'PaymentMethod',
                           'StreamingMovies', 'OnlineBackup', 'SeniorCitizen', 'MultipleLines',
                           'Dependents', 'StreamingTV', 'Partner', 'gender', 'PhoneService', 'DeviceProtection']
 
    # Scale numeric features, one-hot encode categorical features
    preprocessor = ColumnTransformer(transformers=[(
        'num', StandardScaler(),
        numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])
    
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier', SGDClassifier(loss='log'))])
    
    num_features_type_map = {feature: 'float64' for feature in numeric_features}
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)
    
    X_train = df_train.drop('Churn', axis=1)
    y_train = df_train['Churn']
    
    # Set parameters of the model and fit
    pipeline.set_params(classifier__alpha=0.0005, classifier__max_iter=250)
    pipeline.fit(X_train, y_train)
    
    # Save the model locally
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
        
    subprocess.check_call(
        ["gsutil", "cp", model_filename, AIP_MODEL_DIR], stderr=sys.stdout
    )
    print(f"Saved model in: {AIP_MODEL_DIR}")

if __name__ == '__main__':
    fire.Fire(train_evaluate)
