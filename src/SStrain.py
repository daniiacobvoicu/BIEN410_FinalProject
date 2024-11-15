import numpy as np
import json

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

def extract_features(sequence, helix_preferring, window_size):
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

def train_mlp(X, y, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.05, epochs=10000):
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
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    window_size = 8
    hidden1_size = 32
    hidden2_size = 32
    output_size = 1
    
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
    
    X = np.vstack(X)
    y = np.concatenate(y).reshape(-1, 1)  # Reshape y for proper matrix operations

    # Train MLP
    parameters = train_mlp(X, y, input_size=X.shape[1], hidden1_size=hidden1_size, hidden2_size=hidden2_size, output_size=output_size)

    # Save parameters
    save_parameters(parameters)

if __name__ == "__main__":
    main()
