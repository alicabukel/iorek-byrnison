
import os 
import subprocess
import datetime
import fire

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # 27 input features
        self.h1 = nn.Linear(48, 64) 
        self.h2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.h1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.h2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        
        return x

def binary_acc(y_pred, y_true):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]

def train_evaluate(training_dataset_path, validation_dataset_path, batch_size, num_epochs):
    
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    
    # Read in train/validation data and concat 
    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)
    df = pd.concat([df_train, df_validation])

    categorical_features =  ['SeniorCitizen', 'Contract', 'TechSupport', 'OnlineSecurity',
                           'InternetService', 'PaperlessBilling', 'PaymentMethod',
                           'StreamingMovies', 'OnlineBackup', 'SeniorCitizen', 'MultipleLines',
                           'Dependents', 'StreamingTV', 'Partner', 'gender', 'PhoneService', 'DeviceProtection']
    target='Churn'

    # One-hot encode categorical variables 
    df = pd.get_dummies(df,columns=categorical_features)

    df[target] = df[target].apply(lambda x: 0 if x=='Yes' else 1)

    # Split features and labels into 2 different vars
    X_train = df.loc[:, df.columns != target]
    y_train = np.array(df[target])

    # Normalize features 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Training data
    train_data = TrainData(torch.FloatTensor(X_train), 
                           torch.FloatTensor(y_train))

    # Use torch DataLoader to feed data to model 
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, drop_last=True)

    # Instantiate model 
    model = BinaryClassifier()
    
    # Loss is binary crossentropy w/ logits. Must manually implement sigmoid for inference
    criterion = nn.BCEWithLogitsLoss()
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for e in range(1, num_epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        print(f'Epoch {e}: Loss = {epoch_loss/len(train_loader):.5f} | Acc = {epoch_acc/len(train_loader):.3f}')

    # Save the model locally
    model_filename='model.pt'
    torch.save(model.state_dict(), model_filename)

    #EXPORT_PATH = os.path.join(AIP_MODEL_DIR, 'savedmodel')
    EXPORT_PATH = os.path.join(AIP_MODEL_DIR)

    # Copy the model to GCS
    gcs_model_path = '{}/{}'.format(EXPORT_PATH, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path])
    print('Saved model in: {}'.format(gcs_model_path))
    
if __name__ == '__main__':
    fire.Fire(train_evaluate)
