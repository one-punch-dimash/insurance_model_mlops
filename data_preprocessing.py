#Utilities for data load and preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def data_load(raw_data_path):
    
    df = pd.read_csv(raw_data_path, encoding = 'utf-8')
    
    return df
    
def data_preprocessing(df):
    
    df.loc[((df['smoker'] == 'yes'),'smoker')]= '1'
    df.loc[((df['smoker'] == 'no'),'smoker')]= '0'
    df['smoker'] = df['smoker'].astype(int)

    onehot_encoding_columns = ['sex', 'region']
    columns_to_normalize = ['age', 'bmi', 'children']

    for i in onehot_encoding_columns:
        onehot_encoder = OneHotEncoder()
        onehot_encoded = onehot_encoder.fit_transform(df[[i]]).toarray()
        onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out([i]))

        df = pd.concat([df, onehot_encoded_df], axis=1)
        df.drop([i], axis = 1, inplace = True)
        
    for i in columns_to_normalize:
        df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
        
    df.drop(['sex_female'], axis = 1, inplace = True)
    
    #Normalization 
    
    scaler = MinMaxScaler()

    X, y = df.drop(['charges'],axis = 1), df['charges']
    y_normalized = scaler.fit_transform(y.values.reshape(-1, 1))

    # Convert X to a NumPy array
    X_array = X.values

    # Convert X_array and y_normalized to tensors
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    input_size = X_train.shape[1]
    
    return X_train, X_test, y_train, y_test, scaler, input_size
    
