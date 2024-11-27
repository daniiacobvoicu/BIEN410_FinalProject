'''
SSpred.py
===================================================
Python script that makes a alpha-helix secondary structure prediction for a sequence file
===================================================
This script contains 3 functions:
	sigmoid(): Runs a sigmoid function
    relu(): Runs a ReLU function
    load_parameters(): this will call the parameters file and open it as an array of values to be used for the predictions
    one_hot_encode_amino_acids(): converts singular amino acids to one-hot vectors
    extract_features(): Uses a sliding window to extract features around the central amino acid, taking into accound window_size
    predict(): this function is a forward pass through the MLP network using the loaded parameters from training
    main(): Calls all the functions, and outputs things to the appropriate output file. 
'''

import numpy as np
import json

# Amino acid mapping for one-hot encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def load_parameters(filepath="parameters.txt"):
    """Load trained parameters from the JSON file."""
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    # Convert lists back to numpy arrays
    return {key: np.array(val) for key, val in parameters.items()}

def one_hot_encode_amino_acid(aa):
    """Convert a single amino acid to a one-hot vector."""
    vector = np.zeros(len(AMINO_ACIDS))
    if aa in AA_TO_INDEX:
        vector[AA_TO_INDEX[aa]] = 1
    return vector

def extract_features(sequence, window_size):
    """Extract features using one-hot encoding for a sliding window."""
    features = []
    num_amino_acids = len(AMINO_ACIDS)
    for i in range(len(sequence)):
        feature = []
        for j in range(-window_size, window_size + 1):
            if i + j < 0 or i + j >= len(sequence):
                # Padding with zeros for out-of-bound indices
                feature.extend(np.zeros(num_amino_acids))
            else:
                feature.extend(one_hot_encode_amino_acid(sequence[i + j]))
        features.append(feature)
    return np.array(features)

def predict(sequence, parameters, window_size):
    """Predict the secondary structure for a given sequence."""
    # Extract features for the sequence
    X = extract_features(sequence, window_size)
    
    # Forward pass
    W1, b1, W2, b2, W3, b3 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"], parameters["W3"], parameters["b3"]

    # Input to first hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # First hidden layer to second hidden layer
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    # Second hidden layer to output layer
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    # Predict helix or non-helix based on threshold
    predictions = ['H' if p >= 0.5 else '-' for p in A3.flatten()]
    return ''.join(predictions)

def main():
    window_size = 6

    # Load the trained parameters
    parameters = load_parameters("parameters.txt")

    # Open input file and make predictions
    with open("../input_file/infile.txt", 'r') as infile, open("../output_file/outfile.txt", 'w') as outfile:
        lines = infile.readlines()
        
        for i in range(0, len(lines), 2):
            description = lines[i].strip()
            sequence = lines[i+1].strip()
            prediction = predict(sequence, parameters, window_size)
            
            # Write predictions to the output file
            outfile.write(f"{description}\n{sequence}\n{prediction}\n")

if __name__ == "__main__":
    main()
