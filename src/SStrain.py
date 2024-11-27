'''
SStrain.py
===================================================
Python script that trains on training data to create weights in a parameters file for alpha helix prediction
===================================================
This script contains 3 functions:
	sigmoid(): Runs a sigmoid function
    sigmoid(): Derivative of the sigmoid function for the backpropagation
    relu(): Runs a ReLU function
    relu_derivative: Derivative of the ReLU function for the backpropagation
    cross_entropy_loss(): Defines the cross entropy loss function used during gradient descent to adjust weights
    load_training_data(): this will call the training data file and seperate it into sequence, and structure. 
    one_hot_encode_amino_acids(): converts singular amino acids to one-hot vectors
    extract_features(): Uses a sliding window to extract features around the central amino acid, taking into accound window_size
    initialize_parameters(): Using a simplified version of a the xavier initilization to set up the weights
    forward_pass(): Passes the feature vector through the 2 hidden layers, and to the output layer
    backward_pass():Performs gradient descent to output updated weights
    update_parameters(): The weights are adjusted following the backwards pass here, to be used for the next forward pass
    train_mlp(): This iterates the forward and backward pass for several iterations and computes the loss to minimize it. 
    save_parameters(): Prints the weights to the parameter file
    main(): Helps run the entire code, and perform the correct operations
    predict(): this function is a forward pass through the MLP network using the loaded parameters from training
    main(): Calls all the functions, and outputs things to the appropriate output file. 
'''

import numpy as np
import json
import time

# Amino acid mapping for one-hot encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

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

def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    # Xavier initialization for weights
    W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(1 / input_size)
    b1 = np.zeros(hidden1_size)
    W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(1 / hidden1_size)
    b2 = np.zeros(hidden2_size)
    W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(1 / hidden2_size)
    b3 = np.zeros(output_size)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

def forward_pass(X, parameters):
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

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache

def backward_pass(X, y, cache, parameters):
    W1, b1, W2, b2, W3, b3 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"], parameters["W3"], parameters["b3"]
    Z1, A1, Z2, A2, Z3, A3 = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"], cache["Z3"], cache["A3"]

    # Gradients for output layer
    dZ3 = A3 - y
    dW3 = np.dot(A2.T, dZ3) / y.size
    db3 = np.sum(dZ3, axis=0) / y.size

    # Gradients for second hidden layer
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / y.size
    db2 = np.sum(dZ2, axis=0) / y.size

    # Gradients for first hidden layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / y.size
    db1 = np.sum(dZ1, axis=0) / y.size

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    parameters["W3"] -= learning_rate * gradients["dW3"]
    parameters["b3"] -= learning_rate * gradients["db3"]
    return parameters

def train_mlp(X, y, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.05, epochs=3000):
    parameters = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)

    for epoch in range(epochs):
        # Forward pass
        A3, cache = forward_pass(X, parameters)

        # Compute loss
        loss = cross_entropy_loss(y, A3)

        # Backward pass
        gradients = backward_pass(X, y, cache, parameters)

        # Update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)

        # Print loss periodically
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return parameters

def save_parameters(parameters, filepath="parameters.txt"):
    # Convert parameters to lists for JSON serialization
    serializable_parameters = {key: val.tolist() for key, val in parameters.items()}
    with open(filepath, 'w') as f:
        json.dump(serializable_parameters, f)

def main():
    start_time = time.time() #track the time

    window_size = 6
    hidden1_size = 32
    hidden2_size = 16
    output_size = 1
    
    # Load data
    sequences, structures = load_training_data("../training_data/labels.txt")

    # Extract features and labels
    X = []
    y = []
    for sequence, structure in zip(sequences, structures):
        features = extract_features(sequence, window_size)
        labels = np.array([1 if s == 'H' else 0 for s in structure])
        
        X.append(features)
        y.append(labels)
    
    X = np.vstack(X)
    y = np.concatenate(y).reshape(-1, 1)  # Reshape y for proper matrix operations

    # Train MLP
    input_size = len(AMINO_ACIDS) * (2 * window_size + 1)
    parameters = train_mlp(X, y, input_size, hidden1_size, hidden2_size, output_size)

    # Save parameters
    save_parameters(parameters)

    #calculate and print runtime
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
