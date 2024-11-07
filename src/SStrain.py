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

def gini_impurity(y):
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum(proportions ** 2)

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def best_split(X, y):
    best_gini = float("inf")
    best_index, best_threshold = None, None
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, y_left, _, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gini = (len(y_left) * gini_impurity(y_left) + len(y_right) * gini_impurity(y_right)) / len(y)
            if gini < best_gini:
                best_gini = gini
                best_index, best_threshold = feature_index, threshold
    
    return best_index, best_threshold

def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2):
    num_samples, num_features = X.shape
    if depth >= max_depth or num_samples < min_samples_split or np.all(y == y[0]):
        return int(np.bincount(y).argmax())  # Convert to Python int for JSON compatibility

    feature_index, threshold = best_split(X, y)
    if feature_index is None:
        return int(np.bincount(y).argmax())  # Convert to Python int for JSON compatibility

    left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
    left_subtree = build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split)
    right_subtree = build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split)
    return {"feature_index": int(feature_index), "threshold": float(threshold), "left": left_subtree, "right": right_subtree}

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def save_tree(tree, filepath="parameters.txt"):
    tree = convert_to_serializable(tree)  # Convert tree to JSON-serializable format
    with open(filepath, 'w') as f:
        json.dump(tree, f)

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

def main():
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    window_size = 5
    max_depth = 15
    min_samples_split = 10
    
    # Load training data
    sequences, structures = load_training_data("../training_data/labels.txt")
    
    # Prepare features and labels
    X = []
    y = []
    for sequence, structure in zip(sequences, structures):
        features = extract_features(sequence, helix_preferring, window_size)
        labels = np.array([1 if s == 'H' else 0 for s in structure])
        
        X.append(features)
        y.append(labels)
    
    X = np.vstack(X)
    y = np.concatenate(y)

    # Build and save the tree
    tree = build_tree(X, y, max_depth=max_depth, min_samples_split=min_samples_split)
    save_tree(tree)

if __name__ == "__main__":
    main()
