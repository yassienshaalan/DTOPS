# ############################
# JASM: Joint Aspect Sentiment Document Model 
# ############################
### This code was used to train our model "JASM" that simultaneously models both aspects and sentiments from simple input representations. 
    The code is recommeded to be run on GPU machines. However, can be run on CPU machine, but will be much slower. 
### 1. Prerequisities
- Python 3.5
- Numpy
- scipy
- protobuf
- pickle
- glob
- cudamat
- gzip
- Tensorflowgpu==1.11.0
- Keras
- Scikit-learn
- Matplotlib
- logging
- pandas
- textblob
- nltk
- tqdm
	
### 2. Getting Started
      - Get the data first:
      - Due to supplementary material size restrictions, data can to be requested seperately from yassien@gmail.com
      - Preprocessing of Data 
		- a) Extract text and sentiment representation from raw text reviews
	  		- python .\preprocessing\DataPrep_V2.py
    	   	- b) Divide the training and testing data into smaller chunks so that it can be processed in memory with no overflow
    			- python .\preprocessing\DivideTrainTestFiles.py
           	- c) For compatability purposes convert preprocessed input representations into numpy files
   			- python .\preprocessing\ConvertTextFileToNumpy.py 
    
      - Train the Joint Model unsupervised and fine tune either supervised or unsupervised
  		- Change directries to where you store the preprocessed data by editing all paths in runall_joint_dbm.sh
  		- Train JASM and extract representations using the following command
  		- $ ./runall_dbn.sh

##### This implementation has been tested on Red Hat 7.5, Python 3.5, CUDA 8.0 and tensorflow GPU version 1.11.0
