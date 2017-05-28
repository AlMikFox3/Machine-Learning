# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[: , 3].values

#Handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy="mean", axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enc_x = LabelEncoder()
x[:, 0] = label_enc_x.fit_transform(x[:, 0])
onehotenc_x = OneHotEncoder(categorical_features = [0])
x = onehotenc_x.fit_transform(x).toarray()
label_enc_y = LabelEncoder()
y = label_enc_y.fit_transform(y)

#Splitting into train and test set.
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)

#Feature Scaling - Normalization & Standardization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #no fit as the sc_x object is already fit from the previous step