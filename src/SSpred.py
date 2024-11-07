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

import json
import numpy as np

def load_tree(filepath="parameters.txt"):
    with open(filepath, 'r') as f:
        tree = json.load(f)
    return tree

def extract_features(sequence, helix_preferring, window_size=5):
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

def classify_sample(tree, sample):
    if isinstance(tree, dict):
        feature_index = tree["feature_index"]
        threshold = tree["threshold"]
        if sample[feature_index] <= threshold:
            return classify_sample(tree["left"], sample)
        else:
            return classify_sample(tree["right"], sample)
    else:
        return tree

def predict(sequence, tree, helix_preferring, window_size=2):
    X = extract_features(sequence, helix_preferring, window_size)
    predictions = [classify_sample(tree, sample) for sample in X]
    return ''.join(['H' if p == 1 else '-' for p in predictions])

def main():
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    window_size = 5
    
    # Load tree
    tree = load_tree("parameters.txt")

    # Open input file and make predictions
    with open("../input_file/infile.txt", 'r') as infile, open("../output_file/outfile.txt", 'w') as outfile:
        lines = infile.readlines()
        
        for i in range(0, len(lines), 2):
            description = lines[i].strip()
            sequence = lines[i+1].strip()
            prediction = predict(sequence, tree, helix_preferring, window_size)
            
            # Write predictions to output file
            outfile.write(f"{description}\n{sequence}\n{prediction}\n")

if __name__ == "__main__":
    main()
