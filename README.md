# DTOpS
## DTOpS (v1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Detecting Singleton Spams via Learning Deep Anomalous Temporal Aspect-Sentiment Patterns

### 1. Prerequisities
- Python 3.7
- Numpy==1.9.1
- scipy==0.14
- Tensorflow==1.13.0
- Keras==1.0.6 
- Scikit-learn==1.18.5
- Matplotlib==3.2.2
	
### 2. Getting Started
#### - To run DTOPS you need to do the following:
       - 1) Extracted the aspect-sentiment representation (by running our JASM model. Please refer to JASM directory)
       - 2) Prepare the data in time-series format for training (#samples,#time-step,#features), while for testing (#samples,#time-step,#output). Please refer to generate_time_series.py)
#### - Sample code to run DTOPS on all data (Iteratively cleaning training set, train LSTM, then run RVAE)
```python
	from utilities import load_numpy_arrays,load_labels,load_numpy_arrays_withoutXtrain
	from lstm_runner import run_lstm_model,predict_lstm,load_lstm_trained_model
	from rave_runner import run_rave
	import numpy as np
	from outlier_detection_exp_lstm_rave_new import *
	
	print("Loading data")
	X_train,y_train,X_test,y_test = load_numpy_arrays()
	y_train_labels,y_test_labels = load_labels()
	print("Data loaded")
	lamda_testing = [0.00065,0.00075,0.00085]
	lamda_training=0.00050
	print("Run experiment")
	train_test_dtops_all_data(X_train,y_train,X_test,y_test,y_train_labels,y_test_labels,lamda_testing,lamda_training)
```
#### - Sample code to run DTOPS on already trained LSTM model from previous run or expermient
```python
	
	print("Loading data")
	y_train,X_test,y_test = load_numpy_arrays_withoutXtrain()
	y_train_labels,y_test_labels = load_labels()
	print("Data loaded")
	lamda_testing = [0.00065,0.00075,0.00085]
	print("Run experiment")
	model_num = 3 #the trained model number previously save in trained models directory
	test_trained_model(X_test,y_test,y_train_labels,y_test_labels,lamda_list,model_num)
```
#### - Sample code to compare to supervised models by training DTOP's time-series LSTM model on clean (spam-free) data only, then run RVAE 
```python	
	print("Loading data")
	X_train,y_train,X_test,y_test = load_numpy_arrays()
	y_train_labels,y_test_labels = load_labels()
	print("Data loaded")
	lamda_testing = [0.00065,0.00075,0.00085]
	print("Run experiment")
	train_test_dtops_on_normal_data(X_train,y_train,X_test,y_test,y_train_labels,y_test_labels,lamda_list)
```

#### - Sample code run RVAE directly on any representation 
```python	
	print("Loading data")
	y_train,X_test,y_test = load_numpy_arrays_withoutXtrain()
	y_train_labels,y_test_labels = load_labels()
	print("Data loaded")
	lamda_testing = [0.00065,0.00075,0.00085]
	print("Run experiment")
	test_rvae_only(x_test,y_test_labels,lamda_testing)
```
### 3. License
	DTOPS is only distributed under MIT License Copyright (c) 2020.
	Contact: [Yassien Shaalan](mailto:yassien@gmail.com?subject=[GitHub]%20Requesting%20information%20Source/Data%20DTOpS%20Repo)
### 4. Citation
#### If you use this work, please cite:
  ###### {
  ###### title={Detecting Singleton Spams via Learning Deep Anomalous Temporal Aspect-Sentiment Patterns},
  ###### author={Shaalan,Yassien, Zhang, J., Chan, J., Salehi, M.},
  ###### journal={Still under DMKD review},
  ###### year={2020}
###### }
###### This code has been implemented using Python 3.7, tensorflow version 1.13.0, Keras version 1.0.6 on a Ubuntu 18.04 LTS Linux machine with 4 CPUs and 64 GB of memory. 
###### This repo is still under construction.
