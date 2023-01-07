# Script useful to open files.npy contained into the network parameters folder

import numpy as np
import re

thresholdsFile = "./networkParameters/thresholds.npy"
weightsFile = "./networkParameters/weights.npy"
thresholdsOut = "./networkParameters/thresholdsOut.txt"
weightsOut = "./networkParameters/weightsOut.txt"

# Opening thresholdsFile and writing its values into file thresholdsOut.txt
with open(thresholdsFile, 'rb') as fp:
	thresholds = np.load(fp)

with open(thresholdsOut,'w') as f:
	for el in thresholds:
		for el2 in el:
			f.write(str(el2)+'\n')

# Opening weightsFile and writing its values into file weightsOut.txt
with open(weightsFile, 'rb') as fp:
	weights = np.load(fp)

with open(weightsOut,'w') as f:
	for el in weights:
		string = str(el).replace('[','')
		string = string.replace(']','')
		string = string.replace('\n',' ')
		string = re.sub(' +',' ', string)
		f.write(string+'\n')
