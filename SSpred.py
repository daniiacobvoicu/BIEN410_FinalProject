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
#need to maintain the writeOutput part of the function, and the readInput part of the function. Then I just need to modify SS_random_prediction.
#Perhaps I can avoid having a training file, but for more complex models/weights, maybe I need a more complex model to actually get weights. 

# import random
# import numpy as np

# inputFile 		= "../input_file/infile.txt"
# parameters 		= "parameters.txt"
# predictionFile	= "../output_file/outfile.txt"
# labelsFile = "../training_data/labels.txt"

# def readInput(inputFile):
# 	'''
# 		Read the data in a FASTA format file, parse it into into a python dictionnary 
# 		Args: 
# 			inputFile (str): path to the input file
# 		Returns:
# 			training_data (dict): dictionary with format {name (str):sequence (str)} 
# 	'''
# 	inputData = {}
# 	with open(inputFile, 'r') as f:
# 		while True:
# 			name = f.readline()
# 			seq = f.readline()
# 			if not seq: break
# 			inputData.update({name.rstrip():seq.rstrip()})
	
# 	return inputData

# def training_data(labelsFile):
# 	'''
# 	Reads the labelled data in FASTA format file, 
	
# 	'''

# # def SS_random_prediction(inputData,parameters):
# # 	'''
# # 		Predict between alpha-helix (symbol: H) and non-alpha helix (symbol: -) for each amino acid in the input sequences
# # 		The prediction is random but weighted by the overall fraction of alpha helices in the training data (stored in parameters)
# # 		Args: 
# # 			inputData (dict): dictionary with format {name (str):sequence (str)} 
# # 			parameters (str): path to the file with with parameters obtained from training
# # 		Returns:
# # 			randomPredictions (dict): dictionary with format {name (str):ss_prediction (str)} 
# # 	'''

# # 	with open(parameters, 'r') as f:
# # 		fraction = float(next(f))
	
# # 	randomPredictions = {}
# # 	for name in inputData:
# # 		seq = inputData[name]
# # 		preds=""
	
# # 		for aa in seq:
# # 			preds=preds+random.choices(["H","-"], weights = [fraction,1-fraction])[0]
		
# # 		randomPredictions.update({name:preds})
	
# # 	return randomPredictions

# def writeOutput(inputData,predictions,outputFile):
# 	'''
# 		Writes output file with the predictions in the correct format
# 		Args: 
# 			inputData (dict): dictionary with format {name (str):sequence (str)} 
# 			predictions (dict): dictionary with format {name (str):ss_prediction (str)} 
# 			outputFile (str): path to the output file
# 	'''
# 	with open(outputFile, 'w') as f:
# 		for name in inputData:
# 			f.write(name+"\n")
# 			f.write(inputData[name]+"\n")
# 			f.write(predictions[name]+"\n")

# 	return


# def main():

# 	inputData = readInput(inputFile)
# 	predictions = SS_random_prediction(inputData,parameters)
# 	writeOutput(inputData,predictions,predictionFile)

# if __name__ == '__main__':
# 	main()

import json

def load_parameters(filepath="parameters.txt"):
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    return parameters

def predict(sequence, probabilities, window_size=2):
    helix_preferring = {'M', 'A', 'L', 'E', 'K'}
    prediction = []
    
    for i in range(len(sequence)):
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

        # Calculate probability of H vs. non-H using Bayesian decision rule
        prob_helix = (probabilities['non_helix_preferring'][1] ** counts[0]) * \
                     (probabilities['helix_preferring'][1] ** counts[1])
        prob_not_helix = (probabilities['non_helix_preferring'][0] ** counts[0]) * \
                         (probabilities['helix_preferring'][0] ** counts[1])

        # Apply decision rule
        if prob_helix > prob_not_helix:
            prediction.append('H')
        else:
            prediction.append('-')
    
    return "".join(prediction)

def main():
    probabilities = load_parameters("parameters.txt")
    
    with open("../input_file/infile.txt", 'r') as infile, open("../output_file/outfile.txt", 'w') as outfile:
        lines = infile.readlines()
        
        for i in range(0, len(lines), 2):
            description = lines[i].strip()
            sequence = lines[i+1].strip()
            prediction = predict(sequence, probabilities)
            
            outfile.write(f"{description}\n{sequence}\n{prediction}\n")

if __name__ == "__main__":
    main()

