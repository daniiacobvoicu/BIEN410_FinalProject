'''
SStrain.py
===================================================
Python script that computes the fraction of amino-acids labeled as alpha helix in the training data
("../training_data/labels.txt") and writes the fraction to "parameters.txt".
===================================================
This script contains 2 functions:
	readInput(): A function that reads in all labels from the training data and computes the fraction of alpha helices ('H')
	writeOutput(): A function that write the fraction of alpha helix in the training data to "parameters.txt"
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

def calculate_probabilities(sequences, structures, window_size=20):
    amino_acid_counts = {}
    helix_counts = {}
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    
    for sequence, structure in zip(sequences, structures):
        for i in range(len(sequence)):
            amino_acid = sequence[i]
            is_helix = structure[i] == 'H'
            
            if amino_acid not in amino_acid_counts:
                amino_acid_counts[amino_acid] = [0, 0]
            amino_acid_counts[amino_acid][1 if is_helix else 0] += 1

            for j in range(-window_size, window_size + 1):
                if i + j < 0 or i + j >= len(sequence) or j == 0:
                    continue
                neighbor = sequence[i + j]
                if neighbor not in helix_counts:
                    helix_counts[neighbor] = [0, 0]
                helix_counts[neighbor][1 if is_helix else 0] += 1

    probabilities = {}
    for aa, counts in amino_acid_counts.items():
        total = sum(counts)
        if total > 0:
            probabilities[aa] = [counts[0] / total, counts[1] / total]
    
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

