import numpy as np

from mnist import loadDataset
from inputInterface import imgToSpikeTrain
from outputInterface import computePerformance

# Time step duration in milliseconds
dt = 0.1

# Spikes trains duration in milliseconds
trainDuration = 350

# Number of computation steps
computationSteps = int(trainDuration / dt)

# Normalization of the input pixels' values
inputIntensity = 2.

# Number of images after which the accuracy is evaluated
updateInterval = 100

# Network shape
N_layers = 1
N_neurons = [400]
N_inputs = 784

# NumPy default random generator.
rng = np.random.default_rng()

# Mnist test dataset
images = "./mnist/t10k-images-idx3-ubyte"
labels = "./mnist/t10k-labels-idx1-ubyte"

# File containing the label associated to each neuron in the output layer
assignmentsFile = "./networkParameters/assignments.npy"

inputSpikesFilename = "inputSpikes.txt"
outputCountersFilename = "outputCounters.txt"

accuracies = []

# Initialize history of spikes
countersEvolution = np.zeros((updateInterval, N_neurons[-1]))

# Load the assignments from file
with open(assignmentsFile, 'rb') as fp:
    assignments = np.load(fp)

# Import dataset
imgArray, labelsArray = loadDataset(images, labels)

numberOfCycles = 301

# Loop over the whole dataset
for i in range(numberOfCycles):

    print("\nIteration: ", i+1)

    # Translate each pixel into a sequence of spikes
    spikesTrains = imgToSpikeTrain(imgArray[i], dt, computationSteps, inputIntensity, rng)

    # ----------------------------------------------------------------------
    # Scrivere l'array numpy su file nel formato che vi viene più comodo.
    # L'array è formato da 3500 righe, una per ogni step temporale, e 784
    # colonne, una per ogni ingresso.

    with open(inputSpikesFilename, "w") as fp:
        for step in spikesTrains:
            fp.write(str(list(step.astype(int)))
                     [1:-1].replace(",", "").replace(" ", ""))
            fp.write("\n")
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Lanciare il vostro script Rust.
    #
    import subprocess as sp

    rustScript = "../target/debug/pds_snn"

    sp.run(rustScript)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Leggere da file i contatori di uscita e convertirli in un vettore
    # numpy. Qui di seguito uso un vettore fisso che mi serve nelle funzioni
    # successive.
    outputCounters = np.zeros(N_neurons[-1]).astype(int)

    with open(outputCountersFilename, "r") as fp:
        j = 0
        for line in fp:
            outputCounters[j] = int(line)
            j += 1
    # ----------------------------------------------------------------------

    countersEvolution[i % updateInterval] = outputCounters

    accuracies = computePerformance(i, updateInterval, countersEvolution, labelsArray, assignments, accuracies)

