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

import json

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

def calculate_probabilities(sequences, structures, window_size=2):
    helix_counts = {0: [0, 0], 1: [0, 0]}  # Counts for [non-helix, helix]
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    
    for sequence, structure in zip(sequences, structures):
        for i in range(len(sequence)):
            is_helix = structure[i] == 'H'
            counts = [0, 0]  # [non-helix-preferring, helix-preferring]

            # Calculate counts in the window
            for j in range(-window_size, window_size + 1):
                if i + j < 0 or i + j >= len(sequence) or j == 0:
                    continue
                neighbor = sequence[i + j]
                if neighbor in helix_preferring:
                    counts[1] += 1
                else:
                    counts[0] += 1
            
            # Increment helix_counts based on whether the central residue is in a helix
            helix_counts[0][1 if is_helix else 0] += counts[0]
            helix_counts[1][1 if is_helix else 0] += counts[1]

    # Calculate probabilities
    probabilities = {
        'non_helix_preferring': [
            helix_counts[0][0] / (helix_counts[0][0] + helix_counts[0][1]),
            helix_counts[0][1] / (helix_counts[0][0] + helix_counts[0][1])
        ],
        'helix_preferring': [
            helix_counts[1][0] / (helix_counts[1][0] + helix_counts[1][1]),
            helix_counts[1][1] / (helix_counts[1][0] + helix_counts[1][1])
        ]
    }
    
    return probabilities

def save_parameters(parameters, filepath="parameters.txt"):
    with open(filepath, 'w') as f:
        json.dump(parameters, f)

def main():
    sequences, structures = load_training_data("../training_data/labels.txt")
    probabilities = calculate_probabilities(sequences, structures)
    save_parameters(probabilities, "parameters.txt")

if __name__ == "__main__":
    main()

