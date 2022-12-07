import numpy as np


from mnist import loadDataset
from inputInterface import imgToSpikeTrain
from outputInterface import computePerformance


# Time step duration in milliseconds
dt = 0.1

# Spikes trains duration in milliseconds
trainDuration = 350

# Number of computation steps
computationSteps = int(trainDuration/dt)

# Normalization of the input pixels' values
inputIntensity = 2.

# Number of images after which the accuracy is evaluated
updateInterval = 100

# Network shape
N_layers = 1
N_neurons = [400]
N_inputs = 784



# File containing the label associated to each neuron in the output layer
assignmentsFile = "./networkParameters/assignments.npy"
thresholdsFile = "./networkParameters/thresholds.npy"
weightsFile = "./networkParameters/weights.npy"
assignmentsOut = "./networkParameters/assignmentsOut.txt"
thresholdsOut = "./networkParameters/thresholdsOut.txt"
weightsOut = "./networkParameters/weightsOut.txt"

# Load the assignments from file	
with open(assignmentsFile, 'rb') as fp:
	assignments = np.load(fp)


with open(assignmentsOut,'w') as f:
	for el in assignments:
		f.write(str(el)+';'+'\n')
		
	
with open(thresholdsFile, 'rb') as fp:
	thresholds = np.load(fp)

with open(thresholdsOut,'w') as f:
	for el in thresholds:
		for el2 in el:
			f.write(str(el2)+';'+'\n')

with open(weightsFile, 'rb') as fp:
	weights = np.load(fp)

with open(weightsOut,'w') as f:
	for el in weights:
		string = str(el).replace('[','')
		string = string.replace(']','')
		string = string.replace('\n',' ')
		f.write(string+';'+'\n')
