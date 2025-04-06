#-------------------------------------------------------------------------
# AUTHOR: Milosz Kryzia
# FILENAME: perceptron.py
# SPECIFICATION: This program trains and tests both a single and multi-layer perceptron model with data representing handwritten digits, finding 
# which hyperparameters result in the highest accuracy
# FOR: CS 4210- Assignment #3
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_acc = 0
highest_mlp_acc = 0

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        perceptron = [True, False]

        for p in perceptron: #iterates over the algorithms

            #Create a Neural Network classifier
            if p:
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=[25], shuffle=shuffle, max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #for accuracy calculation
            t = 0
            f = 0

            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    t += 1
                else:
                    f += 1
            
            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            accuracy = t / (t+f)
            if p:
                if accuracy > highest_perceptron_acc:
                    highest_perceptron_acc = accuracy
                    print(f'Highest Perceptron accuracy so far: {accuracy}, Parameters: learning_rate={learning_rate}, shuffle={shuffle}')
            else:
                if accuracy > highest_mlp_acc:
                    highest_mlp_acc = accuracy
                    print(f'Highest MLP accuracy so far: {accuracy}, Parameters: learning_rate={learning_rate}, shuffle={shuffle}')