
import os 
import sys
import subprocess
import datetime
import fire
import pickle 

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]

def train_evaluate(training_dataset_path, validation_dataset_path,max_depth,n_estimators):
    
    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)
    df = pd.concat([df_train, df_validation])

    categorical_features = ['SeniorCitizen', 'Contract', 'TechSupport', 'OnlineSecurity',
                           'InternetService', 'PaperlessBilling', 'PaymentMethod',
                           'StreamingMovies', 'OnlineBackup', 'SeniorCitizen', 'MultipleLines',
                           'Dependents', 'StreamingTV', 'Partner', 'gender', 'PhoneService', 'DeviceProtection']
    target='Churn'

    # One-hot encode categorical variables 
    df = pd.get_dummies(df,columns=categorical_features)

    # Change label to 0 if <=50K, 1 if >50K
    df[target] = df[target].apply(lambda x: 0 if x=='Yes' else 1)

    # Split features and labels into 2 different vars
    X_train = df.loc[:, df.columns != target]
    y_train = np.array(df[target])

    # Normalize features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    grid = {
        'max_depth': int(max_depth),
        'n_estimators': int(n_estimators)
    }
    
    model = XGBClassifier()
    model.set_params(**grid)
    model.fit(X_train,y_train)
    
    # Save the model locally
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
        
    subprocess.check_call(
        ["gsutil", "cp", model_filename, AIP_MODEL_DIR], stderr=sys.stdout
    )
    print(f"Saved model in: {AIP_MODEL_DIR}")
    
if __name__ == '__main__':
    fire.Fire(train_evaluate)
