'''
SStrain.py
===================================================
Python script that computes the fraction of amino-acids labeled as alpha helix in the training data based on a moving window
("../training_data/labels.txt") and writes the fraction to "parameters.txt".
===================================================
This script contains 2 functions:
	load_training_data(): A function that reads in all labels from the training data and returns the sequences, and the structures seperately 
    calculate_probabilities(): A function that calculates the probabilities of an H
	Ca(): A function that write the fraction of alpha helix in the training data to "parameters.txt"
'''
##ADJUST

import numpy as np
import json

#definining initial logistic regression functions
#sigmoid function to convert to a probability between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Cross-entropy loss to figure out the difference between predicted and actual labels
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#load up input data
def load_training_data(labels_file):
    sequences = []
    structures = []
    
    with open(labels_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            sequence = lines[i+1].strip()
            structure = lines[i+2].strip()
            sequences.append(sequence)
            structures.append(structure)
    
    return sequences, structures

def extract_features(sequence, helix_preferring, window_size=2):
    # Initialize feature array
    features = []
    
    for i in range(len(sequence)):
        feature = []
        for j in range(-window_size, window_size + 1):
            if i + j < 0 or i + j >= len(sequence):
                feature.append(0)  # Padding for out-of-bounds
            else:
                feature.append(1 if sequence[i + j] in helix_preferring else 0)
        
        features.append(feature)
    
    return np.array(features)

def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    # Initialize weights
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        # Linear model
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Calculate gradients
        dw = np.dot(X.T, (y_pred - y)) / y.size
        db = np.sum(y_pred - y) / y.size

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Optional: Print loss for monitoring training
        if epoch % 100 == 0:
            loss = cross_entropy_loss(y, y_pred)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, bias

def save_parameters(weights, bias, filepath="parameters.txt"):
    parameters = {
        'weights': weights.tolist(),
        'bias': bias
    }
    with open(filepath, 'w') as f:
        json.dump(parameters, f)

def main():
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    window_size = 2
    
    # Load data
    sequences, structures = load_training_data("../training_data/labels.txt")

    # Extract features and labels
    X = []
    y = []
    for sequence, structure in zip(sequences, structures):
        features = extract_features(sequence, helix_preferring, window_size)
        labels = np.array([1 if s == 'H' else 0 for s in structure])
        
        X.append(features)
        y.append(labels)
    
    X = np.vstack(X)  # Combine all features into one array
    y = np.concatenate(y)  # Combine all labels into one array

    # Train logistic regression model
    weights, bias = train_logistic_regression(X, y)

    # Save parameters
    save_parameters(weights, bias)

if __name__ == "__main__":
    main()


