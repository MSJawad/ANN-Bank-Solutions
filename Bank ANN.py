# -*- coding: utf-8 -*-
"""
Created on %(14/5/2018)s

@author: %(msjawad)
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set

# ANN and improving the ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Initializing the ANN
classifier = Sequential()


# adding input layer and hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim = 11))
classifier.add(Dropout(p = 0.1))


#add second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu"))
classifier.add(Dropout(p = 0.1))


# add output layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))


# Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])


# fitting the ANN onto training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

hw_test = np.array([[600, 'France' , 'Male' , 40, 3 , 60000, 2, 1, 1, 50000]])
hw_test[:, 1] = labelencoder_X_1.transform(hw_test[:, 1])
hw_test[:, 2] = labelencoder_X_2.transform(hw_test[:, 2])
hw_test = onehotencoder.transform(hw_test).toarray()
hw_test = hw_test[:, 1:]

hw_test = sc.transform(hw_test)

new_pred = classifier.predict(hw_test)
new_pred = new_pred > 0.5

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
   from keras.models import Sequential
   from keras.layers import Dense
   from keras.layers import Dropout
    
   classifier = Sequential()
   
   classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
   classifier.add(Dropout(p = 0.1))
   classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
   classifier.add(Dropout(p = 0.1))
   classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))   
   classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
   
   return classifier
    

classifier = KerasClassifier(build_fn= build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

mean = 0
for x in accuracies:
    mean = mean + x
    
mean = mean/10


# tuning the ANN parameters so that they are better suited(number of neurons, batch size, etc)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



def build_classifier(optimizer):
   from keras.models import Sequential
   from keras.layers import Dense
   from keras.layers import Dropout
    
   classifier = Sequential()
   
   classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
   classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
   classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))   
   classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
   
   return classifier
    

classifier = KerasClassifier(build_fn = build_classifier)

parameters= {'batch_size': [25,32],
             'epochs': [100,500],
             'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy', cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_






