import numpy as np

def computePerformance(currentIndex, updateInterval, countersEvolution, 
			labels, assignments, accuracies):

	'''
	Compute the network performance.

	INPUT:	

		1) currentIndex: index of the current image.

		2) updateInterval: number of images after which the performance
		is computed.

		3) countersEvolution: two-dimensional NumPy array containing the
		history of the spikes counters in the last "updateInterval"
		cycles. One row for each training step. One column for each
		element in the output layer.

		4) labels: NumPy array containing all the labels.

		5) assignments: NumPy array containing one label assignment for
		each output neuron.

		6) accuracies: list of strings containing the history of the
		accuracy.

	OUTPUT:

		accuracies: updated list of strings containing the history of the
		accuracy.

	'''


	# End of update interval?
	if currentIndex % updateInterval == 0 and currentIndex > 0:

		# Initialize the maximum count to 0
		maxCount = np.zeros(updateInterval)

		# Initialize the output classification
		classification = -1*np.ones(updateInterval, dtype=np.int32)

		labelsSequence = labels[currentIndex - updateInterval :
				currentIndex]


		for label in range(10):

			# Add the spikes count associated to the current label
			spikesCount = np.sum(countersEvolution[:, assignments ==
					label], axis = 1)


			# Find where the spikes count is greater than the maximum
			whereMaxSpikes = spikesCount > maxCount

			# Associate the instants to the current label
			classification[whereMaxSpikes] = label

			# Update the maximum number of spikes for the label
			maxCount[whereMaxSpikes] = spikesCount[whereMaxSpikes]

		# print computed labels
		print(classification)

		# Compute the accuracy and add it to the list of accuracies
		accuracies = updateAccuracy(classification, labelsSequence, accuracies)

	return accuracies




def updateAccuracy(classification, labelsSequence, accuracies):

	'''
	Compute the accuracy and add it to the list of accuracies.

	INPUT:

		1) classification: NumPy array containing the history of the
		classification performed by the network in the last
		"updateInterval" cycles

		2) labelsSequence: NumPy array containing the history of the
		labels in the last "updateInterval" cycles.

		3) accuracies: list of strings containing the history of the
		accuracy.

	OUTPUT:

		accuracies: updated list of strings containing the history of the
		accuracy.

	'''

	# Number of instants in which the classification is equal to the label
	correct = np.where(classification == labelsSequence)[0].size

	# Compute the percentage of accuracy and add it to the list
	accuracies += ["{:.2f}".format(correct/classification.size*100) + "%"]
	
	# Print the accuracy
	accuracyString = "\nAccuracy: " + str(accuracies) + "\n"

	print(accuracyString)

	return accuracies
