# DTOpS
## DTOpS (v1.0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Detecting Singleton Spams in Reviews via Learning Deep Anomalous Temporal Aspect-Sentiment Patterns

Customer reviews are an essential source of information to consumers. 
Meanwhile, opinion spams spread widely and the detection of spam reviews becomes critically important for ensuring the integrity of the echo system of online reviews.
Singleton spam reviews – one-time reviews – have spread widely of late as spammers can create multiple accounts to purposefully cheat the system. 
Most available techniques fail to detect this cunning form of malicious reviews, mainly due to the scarcity of behaviour trails left behind by singleton spammers. 
Available approaches also require extensive feature engineering, expensive manual annotation and are less generalizable. 
Based on our thorough study of spam reviews, it was found that genuine opinions are usually directed uniformly towards important aspects of entities. 
In contrast, spammers attempt to counter the consensus towards these aspects while covering their malicious intent by adding more text but on less important aspects. 
Additionally, spammers usually target specific time periods along products' lifespan to cause maximum bias to the public opinion. Based on these observations, we present an unsupervised singleton spam review detection model that runs in two steps. 
Unsupervised deep aspect-level sentiment model employing deep Boltzmann machines (DBMs) first learns fine-grained opinion representations from review texts. 
Then, an LSTM network is trained on opinion learned representation to track the evolution of opinions through the fluctuation of sentiments in a temporal context, followed by the application of a Robust Variational Autoencoder to identify spam instances. Experiments on three benchmark datasets widely used in the literature showed that our approach outperforms strong state-of-the-art baselines.

Here is a link to the paper on springer [Detecting singleton spams in reviews via learning deep anomalous temporal aspect-sentiment patterns](https://link.springer.com/article/10.1007%2Fs10618-020-00725-5).
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
	lamda_testing = [0.00065,0.00075,0.00085] #just sample values
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
	lamda_testing = [0.00065,0.00075] #just sample values
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
	lamda_testing = [0.00065,0.00075,0.00085] #just sample values
	print("Run experiment")
	train_test_dtops_on_normal_data(X_train,y_train,X_test,y_test,y_train_labels,y_test_labels,lamda_list)
```

#### - Sample code run RVAE directly on any representation 
```python	
	print("Loading data")
	y_train,X_test,y_test = load_numpy_arrays_withoutXtrain()
	y_train_labels,y_test_labels = load_labels()
	print("Data loaded")
	lamda_testing = [0.00065,0.00075,0.00085] #just sample values
	print("Run experiment")
	test_rvae_only(x_test,y_test_labels,lamda_testing)
```
### 3. License
-	DTOPS is only distributed under Apache-2.0 License Copyright (c) 2020.
-	Contact: [Yassien Shaalan](mailto:yassien@gmail.com?subject=[GitHub]%20Requesting%20information%20Source/Data%20DTOpS%20Repo)
### 4. Citation
#### If you use this work, please cite:
{
```  
  title={Detecting Singleton Spams via Learning Deep Anomalous Temporal Aspect-Sentiment Patterns},
  author={Shaalan,Yassien, Zhang, X., Chan, J., Salehi, M.},
  journal={Data Mining and Knowledge Discovery (DMKD)},
  pages = {1-56},
  url={https://doi.org/10.1007/s10618-020-00725-5},
  year={2021}
```
}

###### This code has been implemented using Python 3.7, tensorflow version 1.13.0, Keras version 1.0.6 on a Ubuntu 18.04 LTS Linux machine with 4 CPUs and 64 GB of memory. 
###### This repo is still under construction.
