# Python 3 script to edit for this project. 
# Note 1: Do not change the name of this file
# Note 2: Do not change the location of this file within the BIEN410_FinalProject package
# Note 3: This file can only read in "../input_file/input_file.txt" and "parameters.txt" as input
# Note 4: This file should write output to "../output_file/outfile.txt"
# Note 5: See example of a working SSPred.py file in ../src_example folder
'''
SSpred.py
===================================================
Python script that makes a alpha-helix secondary structure prediction for a sequence file
===================================================
This script contains 3 functions:
	readInput(): A function that reads in input from a sequence file
	SS_random_prediction(): A function that makes a random prediction weighted by the fraction of alpha helices in the training data
	writeOutput(): A function that write a prediction to an output file 
'''

import numpy as np
import json

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def load_parameters(filepath="parameters.txt"):
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    # Convert lists back to numpy arrays
    return {key: np.array(val) for key, val in parameters.items()}

def extract_features(sequence, helix_preferring, window_size=2):
    features = []
    for i in range(len(sequence)):
        feature = []
        for j in range(-window_size, window_size + 1):
            if i + j < 0 or i + j >= len(sequence):
                feature.append(0)
            else:
                feature.append(1 if sequence[i + j] in helix_preferring else 0)
        features.append(feature)
    return np.array(features)

def predict(sequence, parameters, helix_preferring, window_size=2):
    # Extract features for the sequence
    X = extract_features(sequence, helix_preferring, window_size)
    
    # Forward pass
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

    # Input to hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Hidden layer to output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Predict helix or non-helix based on threshold
    predictions = ['H' if p >= 0.5 else '-' for p in A2.flatten()]
    return ''.join(predictions)

def main():
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    window_size = 5

    # Load the trained parameters
    parameters = load_parameters("parameters.txt")

    # Open input file and make predictions
    with open("../input_file/infile.txt", 'r') as infile, open("../output_file/outfile.txt", 'w') as outfile:
        lines = infile.readlines()
        
        for i in range(0, len(lines), 2):
            description = lines[i].strip()
            sequence = lines[i+1].strip()
            prediction = predict(sequence, parameters, helix_preferring, window_size)
            
            # Write predictions to the output file
            outfile.write(f"{description}\n{sequence}\n{prediction}\n")

if __name__ == "__main__":
    main()

