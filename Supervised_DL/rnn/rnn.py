"""Recurrent Neural Network, using LSTMs, predicting stock price of google 
"""
# Part 1 Data preprocessing 

# importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set 
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timestamps and 1 output 
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
    
    
    
    
# Part 2 Building the RNN



# Part 3 Making the predictions and vizualizing the results 