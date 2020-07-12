from utilities import load_numpy_arrays,load_labels
from lstm_runner import run_lstm_model,predict_lstm,load_lstm_trained_model
from rave_runner import run_rave
import numpy as np
from scipy import spatial
from statistics import  *

def difference(X,y,absolute=0):
     """
        Create a matrix as of difference between two matrices.
        Args:
            X: First matrix (2d numpy array) 
            y: Second matrix (2d numpy array)
        Attributes:
            diff_m: The resultant difference metrix (2d numpy array)
            absolute: Provide absolute difference (0 by default)
     """
    temp = y
    temp = temp.reshape(y.shape[0],y.shape[2])
    diff_m = []
    for i in range(len(temp)):
        row = []
        for j in range(len(temp[i])):
            if absolute==1:
                row.append(abs(temp[i][j]-X[i][j]))
            else:
                row.append(X[i][j]-temp[i][j])
        diff_m.append(row)
    diff_m = np.array(diff_x)

    #mean_diff = np.mean(diff_x, axis = 0)
    #print("mean diff on all arr",mean_diff,mean_diff.shape)
    #print(max(mean_diff),min(mean_diff),mean(mean_diff))

    return diff_x

def combination(X,y):
    temp = y
    temp = temp.reshape(y.shape[0],y.shape[2])
    comb_x = []
    for i in range(len(temp)):
        row = []
        for j in range(len(temp[i])):
            row.append(temp[i][j])
            row.append(X[i][j])
            row.append(X[i][j]-temp[i][j])
        comb_x.append(row)
    comb_x = np.array(comb_x)

    return comb_x
def normalize_matrix(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def training_stage_method_0(X_train,y_train,y_train_labels):
    done = False
    index=1
    suspected_indices=[]
    iter = 0
    X_train_new = X_train
    y_train_new = y_train
    max_iters = 10
    lam_tr= 0.0010
    checked_before = []
    acceptable_suspect = 0
    acceptable_train_rate = 100
    first = 0
    while done!=True and index<max_iters:
        #filtering anomalies from training data
        print("*************************")
        print("Iteration ",iter+1)
        X_train_filtered = []
        y_train_filtered = []
        print("len of checked_before",len(checked_before))
        for i in range(X_train.shape[0]):
            if i not in suspected_indices :#and i not in checked_before:
                X_train_filtered.append(X_train_new[i])
                y_train_filtered.append(y_train_new[i])
                if first!=0:
                    checked_before.append(i)

        if first==0:
            first = 1
        if len(X_train_filtered) <acceptable_train_rate:
            print("Breaking as num training became ",len(X_train_filtered))
            break
        X_train_filtered=np.array(X_train_filtered)
        y_train_filtered = np.array(y_train_filtered)
        print("Filtered shape",X_train_filtered.shape,y_train_filtered.shape)

        print("Filtered shape train",X_train_filtered.shape,y_train_filtered.shape)
        #X_train_new =X_train_filtered
        #y_train_new=y_train_filtered
        '''
        tempo = y_train_filtered
        temp = tempo.reshape(y_train.shape[0],y_train.shape[2])
        print("Run RAVE before LSTM",temp.shape,y_train_labels.shape)
        suspected_indices = run_rave(temp,y_train_labels,lam = 0.0009)
        '''
        
        print("Run LSTM")
        lstm_model = run_lstm_model(X_train_filtered, y_train_filtered, l_s=5, l_p=1,verbose=False,model_num=index)
        xtrain_pred = predict_lstm(lstm_model,X_train_filtered)
        #tempo = y_train
        #temp = tempo.reshape(y_train.shape[0],y_train.shape[2])
        #print("Sanity",xtrain_pred.shape,y_train.shape,temp.shape)
        #suspected_indices = run_rave(temp,y_train_labels,lam = 0.0009)
        #print("Run RAVE",xtrain_pred.shape,y_train_labels.shape)
        #diff_x = y_train_filtered-xtrain_pred
        print("xtrain_pred",xtrain_pred.shape)
        diff_x = differece(xtrain_pred,y_train_filtered)
        print("diff_x",diff_x.shape)
        #print(diff_x[0])
        suspected_indices = run_rave(diff_x,y_train_labels,lam = lam_tr)

        print("suspected_indices now",len(suspected_indices))
        if len(suspected_indices)==acceptable_suspect:#clean
            done = True
            print("I am out after iteration",iter)
            break
        index+=1
        iter+=1
        lam_tr += 0.0005#0.00005
        print("*************************")
        #break
    return lstm_model

def training_stage_method_1(X_train,y_train,y_train_labels):
    done = False
    index=1
    suspected_indices=[]
    iter = 0
    X_train_new = X_train
    y_train_new = y_train
    max_iters = 10
    lam_tr= 0.0010
    checked_before = []
    acceptable_suspect = 0
    acceptable_train_rate = 100
    first = 0
    while done!=True and index<max_iters:
        #filtering anomalies from training data
        print("*************************")
        print("Iteration ",iter+1)
        X_train_filtered = []
        y_train_filtered = []
        print("len of checked_before",len(checked_before))
        for i in range(X_train.shape[0]):
            if i not in suspected_indices :#and i not in checked_before:
                X_train_filtered.append(X_train_new[i])
                y_train_filtered.append(y_train_new[i])
                if first!=0:
                    checked_before.append(i)

        if first==0:
            first = 1
        if len(X_train_filtered) <acceptable_train_rate:
            print("Breaking as num training became ",len(X_train_filtered))
            break
        X_train_filtered=np.array(X_train_filtered)
        y_train_filtered = np.array(y_train_filtered)
        print("Filtered shape",X_train_filtered.shape,y_train_filtered.shape)

        num_tr = int(len(X_train_filtered)*0.7)
        X_train_filtered_tr = X_train_filtered[:num_tr]
        y_train_filtered_tr = y_train_filtered[:num_tr]
        X_train_filtered_ts = X_train_filtered[num_tr:]
        y_train_filtered_ts = y_train_filtered[num_tr:]
        X_train_filtered = X_train_filtered_tr
        y_train_filtered = y_train_filtered_tr
        print("Filtered shape train",X_train_filtered.shape,y_train_filtered.shape)
        #X_train_new =X_train_filtered
        #y_train_new=y_train_filtered
        '''
        tempo = y_train_filtered
        temp = tempo.reshape(y_train.shape[0],y_train.shape[2])
        print("Run RAVE before LSTM",temp.shape,y_train_labels.shape)
        suspected_indices = run_rave(temp,y_train_labels,lam = 0.0009)
        '''
        
        print("Run LSTM")
        lstm_model = run_lstm_model(X_train_filtered, y_train_filtered, l_s=5, l_p=1,verbose=False,model_num=index)
        xtrain_pred = predict_lstm(lstm_model,X_train_filtered_ts)
        #tempo = y_train
        #temp = tempo.reshape(y_train.shape[0],y_train.shape[2])
        #print("Sanity",xtrain_pred.shape,y_train.shape,temp.shape)
        #suspected_indices = run_rave(temp,y_train_labels,lam = 0.0009)
        #print("Run RAVE",xtrain_pred.shape,y_train_labels.shape)
        #diff_x = y_train_filtered-xtrain_pred
        print("xtrain_pred",xtrain_pred.shape)
        diff_x = differece(xtrain_pred,y_train_filtered_ts)
        print("diff_x",diff_x.shape)
        #print(diff_x[0])
        suspected_indices = run_rave(diff_x,y_train_labels,lam = lam_tr)

        print("suspected_indices now",len(suspected_indices))
        if len(suspected_indices)==acceptable_suspect:#clean
            done = True
            print("I am out after iteration",iter)
            break
        index+=1
        iter+=1
        lam_tr += 0.0005#0.00005
        print("*************************")
        #break
    return lstm_model

def training_stage_method_2(X_train,y_train,y_train_labels,lam_tr):
    print("This is training method 2")
    done = False
    suspected_indices=[]
    iter = 1
    X_train_new = X_train
    y_train_new = y_train
    max_iters = 10
    #lam_tr= 0.00070#0.0015 #0.003
    #checked_before = []
    acceptable_suspect = 50
    acceptable_train_rate =  int(0.5*len(X_train))
    first = 1
    num_tr = int(len(X_train_new)*0.7)
    X_train_filtered_tr = X_train_new[:num_tr]
    y_train_filtered_tr = y_train_new[:num_tr]
    y_train_label_tr = y_train_labels[:num_tr]

    X_train_filtered_ts = X_train_new[num_tr:]
    y_train_filtered_ts = y_train_new[num_tr:]
    y_train_label_ts = y_train_labels[num_tr:]

    while done!=True and iter<max_iters:
        #filtering anomalies from training data
        print("lam_tr",lam_tr)
        X_train_filtered = []
        y_train_filtered = []
        #print("len of checked_before",len(checked_before))
        if first == 1:
            X_train_filtered = X_train_filtered_tr
            y_train_filtered = y_train_filtered_tr
            first = 0
        else:
            temp_x = []
            temp_y = []
            for i in range(X_train_filtered_ts.shape[0]):#collect clean instances from previous testing set to add to training
                if i not in suspected_indices :
                    temp_x.append(X_train_filtered_ts[i])
                    temp_y.append(y_train_filtered_ts[i])

            num_tr = int(len(X_train_filtered_tr)*0.7) #split old training into train and test 
            X_train_filtered_ts = X_train_filtered_tr[num_tr:]
            y_train_filtered_ts = y_train_filtered_tr[num_tr:]
            y_train_label_ts = y_train_label_tr[num_tr:]

            X_train_filtered_tr = X_train_filtered_tr[:num_tr]
            y_train_filtered_tr = y_train_filtered_tr[:num_tr]
            y_train_label_tr = y_train_label_tr[:num_tr]
 
            for i in range(X_train_filtered_tr.shape[0]):
                    X_train_filtered.append(X_train_filtered_tr[i])
                    y_train_filtered.append(y_train_filtered_tr[i])
            #the we add the old filtered test as it is timeseries so we put them at the end of the training
            for i in range(len(temp_x)):
                    X_train_filtered.append(temp_x[i])
                    y_train_filtered.append(temp_y[i])
        
        if len(X_train_filtered) <acceptable_train_rate:
            print("Breaking as num training became ",len(X_train_filtered))
            iter-=1
            break
        print("Iteration ",iter)
        X_train_filtered=np.array(X_train_filtered)
        y_train_filtered = np.array(y_train_filtered)
        print("Filtered shape X",X_train_filtered.shape,"Y",y_train_filtered.shape,"label train",y_train_label_tr.shape,"label test",y_train_label_ts.shape)

        print("Run LSTM on other part")
        lstm_model = run_lstm_model(X_train_filtered, y_train_filtered, l_s=5, l_p=1,verbose=False,model_num=iter)
        xest_pred = predict_lstm(lstm_model,X_train_filtered_ts)
        print("LSTM Prediction shape",xest_pred.shape,y_train_filtered_ts.shape)
        #diff_x = differece(lst_pred_x,y_train_filtered_ts)
        diff_x = combination(xest_pred,y_train_filtered_ts)
        print("Run Rave on difference ",diff_x.shape)
        suspected_indices = run_rave(diff_x,y_train_label_ts,lam = lam_tr)
        #print("suspected_indices now",len(suspected_indices))
        if len(suspected_indices)<=acceptable_suspect:#clean
            done = True
            print("I am out after iteration",iter)
            break
        #if iter==2:
        #    print("I am breaking i am telling you this")
        #    break   
        else:
            iter+=1
            lam_tr += (lam_tr*0.10)#0.0001 #0.002#0.00005
            print("*************************")
            #break
    
    return iter
def train_test_dtops_all_data(X_train,y_train,X_test,y_test,y_train_labels,y_test_labels,lamda_list,lam_tr):
     """
        Run DTOPS algorithm on all data (not execluding spam from traing set) on a range of lamdas 
        We print Accuracy, Precsion, Recall & F1 score for each run per lamda in lamda list
        Args:
            X_train: Time-Series training data (3d numpy array) (#samples,#timestep,#features)
            y_train: Target data for time series prediciton (2d numpy array) (#samples,#output)
            X_test: Time-Series testing data (3d numpy array) (#samples,#timestep,#features)
            y_test: Target data for time series prediciton (2d numpy array) (#samples,#output)
            y_train_labels: Training data labels array with values (1: spam, 0:non-spam) (1d numpy array) not used in training just for monitoring
            y_test_labels: Testing data labels array with values (1: spam, 0:non-spam) (1d numpy array)
     """
    #Training an LSTM to predict reviews evolution 
    model_num = training_stage_method_2(X_train,y_train,y_train_labels,lam_tr)
    print("The final model to ",model_num)
    #load already trained lstm model
    lstm_model = load_lstm_trained_model(model_num,X_train.shape[2])
    print("*************************")
    print("Last LSTM after filtering out training data")
    print("Now predict testing testing set",X_test.shape)
    xest_pred = predict_lstm(lstm_model,X_test)
    #diff_x = difference(xest_pred,y_test)
    diff_x = combination(xest_pred,y_test)
    print("Now detecting spam in testing set")
    print("diff_x  for testing",diff_x.shape)
    lam_test = 0.00055
    print("Expected lamda",diff_x.shape[0]*lam_test)
    #lamda_list = np.arange(0.00035,0.0015,0.0001)
    #lamda_list = [0.00070,0.00072,0.00074,0.00076,0.00078,0.00080]
    for lam_test in lamda_list:
        print("Expected lamda",diff_x.shape[0]*lam_test)
        suspected_indices = run_rave(diff_x,y_test_labels,lam =lam_test)# 0.0022)
        #print("suspected_indices final",len(suspected_indices))
        print("*************************")

    #lamda_list = [0.00070,0.00072,0.00074,0.00076,0.00078,0.00080]
    for lam_test in lamda_list:
        print("Expected lamda",diff_x.shape[0]*lam_test)
        suspected_indices = run_rave(diff_x,y_test_labels,lam =lam_test)# 0.0022)
        #print("suspected_indices final",len(suspected_indices))
        print("*************************")
    #suspected_indices = run_rave(diff_x,y_test_labels,lam =lam_test)# 0.0022)
    #print("suspected_indices final",len(suspected_indices))
    #print("*************************")
    #print("looking at preds only not diff")
    #suspected_indices = run_rave(xest_pred,y_test_labels,lam =lam_test)# 0.0022)
    #print("suspected_indices final",len(suspected_indices))
    #print("*************************")
  
    return

def measure_similarity(X_pred,y_test):
    y_test_t = y_test.reshape(y_test.shape[0],y_test.shape[2])
    print("y_test shape", y_test.shape, "X_pred shape",X_pred.shape)
    similarties = []
    for j in range(len(y_test)):
        similarties.append(1 - spatial.distance.cosine(y_test[j], X_pred[j]))
    print("Simlarties", len(similarties),"max", max(similarties),"min", min(similarties))
    print("mean", mean(similarties),"median", median(similarties))

    return
def train_test_dtops_on_normal_data(X_train,y_train,X_test,y_test,y_train_labels,y_test_labels,lamda_list):
    """
        Train LSTM on spam-free data then run run DTOPS on a range of lamdas 
        We print Accuracy, Precsion, Recall & F1 score for each run per lamda in lamda list
        Args:
            X_train: Time-Series training data (3d numpy array) (#samples,#timestep,#features)
            y_train: Target data for time series prediciton (2d numpy array) (#samples,#output)
            X_test: Time-Series testing data (3d numpy array) (#samples,#timestep,#features)
            y_test: Target data for time series prediciton (2d numpy array) (#samples,#output)
            y_train_labels: Training data labels array with values (1: spam, 0:non-spam) (1d numpy array) not used in training just for monitoring
            y_test_labels: Testing data labels array with values (1: spam, 0:non-spam) (1d numpy array)
     """
    print("Train LSTM only on clean data")
    print("Filtering spam data from training")
    X_train_filtered = []
    y_train_filtered = []
    for i in range(X_train.shape[0]):
        if y_train_labels[i]!=1:
             X_train_filtered.append(X_train[i])
             y_train_filtered.append(y_train[i])
    X_train_filtered=np.array(X_train_filtered)
    y_train_filtered = np.array(y_train_filtered)
    print("Filtered after removing spam from trainingshape X",X_train_filtered.shape,"Y",y_train_filtered.shape)
    lstm_model = run_lstm_model(X_train_filtered, y_train_filtered, l_s=5, l_p=1,verbose=False,model_num=0)
    print("Loading saved model")
    lstm_model = load_lstm_trained_model(0,X_test.shape[2])
    print("Predicting Data")
    X_pred = predict_lstm(lstm_model,X_test)
    #measure_similarity(X_pred,y_test)
    #diff_x = difference(X_pred,y_test,absolute=0)
    diff_x = combination(X_pred,y_test)
    print("Now detecting spam in testing set")
    print("diff_x  for testing",diff_x.shape)
    print(diff_x[0])
    '''
    diff_x = normalize_matrix(diff_x,0,1)
    print("diff_x  after normalization",diff_x.shape)
    print(diff_x[0])
    '''
    #lam_test = 0.00075#0.00055#0.0015#0.00055#0.00035#0.00090
    #lamda_list = np.arange(0.00035,0.0015,0.0001)
    #lamda_list = [0.00070,0.00072,0.00074,0.00076,0.00078,0.00080]
    for lam_test in lamda_list:
        print("Expected lamda",diff_x.shape[0]*lam_test)
        suspected_indices = run_rave(diff_x,y_test_labels,lam =lam_test)# 0.0022)
        #print("suspected_indices final",len(suspected_indices))
        print("*************************")
    #print("looking at preds only not diff")
    #suspected_indices = run_rave(xest_pred,y_test_labels,lam =lam_test)# 0.0022)
    #print("suspected_indices final",len(suspected_indices))
    #print("*************************")
    ##########
   
    return
    
def test_trained_model(X_test,y_test,y_train_labels,y_test_labels,lamda_list,model_num):
     """
        Run DTOPS using a previously trained LSTM model on a range of lamdas 
        We print Accuracy, Precsion, Recall & F1 score for each run per lamda in lamda list
        Args:
            X_test: Time-Series testing data (3d numpy array) (#samples,#timestep,#features)
            y_test: Target data for time series prediciton (2d numpy array) (#samples,#output)
            y_train_labels: Training data labels array with values (1: spam, 0:non-spam) (1d numpy array) not used in training just for monitoring
            y_test_labels: Testing data labels array with values (1: spam, 0:non-spam) (1d numpy array)
     """
    print("Loading saved model LSTM_v",str(model_num))
    lstm_model = load_lstm_trained_model(model_num,X_test.shape[2])
    print("Predicting Data")
    X_pred = predict_lstm(lstm_model,X_test)
    #measure_similarity(X_pred,y_test)
    #diff_x = difference(X_pred,y_test,absolute=0)
    print("Preparing Data for RVAE")
    diff_x = combination(X_pred,y_test)
    print("Now detecting spam in testing set")
    print("diff_x  for testing",diff_x.shape)
    print(diff_x[0])
    '''
    diff_x = normalize_matrix(diff_x,0,1)
    print("diff_x  after normalization",diff_x.shape)
    print(diff_x[0])
    '''
    #lam_test = 0.00075#0.00055#0.0015#0.00055#0.00035#0.00090
    
    #lamda_list = np.arange(0.00035,0.0015,0.0001)
    #lamda_list = [0.00070,0.00072,0.00074,0.00076,0.00078,0.00080]
    #lamda_list = [0.000010,0.000011,0.000012,0.000013,0.000014]
    
    for lam_test in lamda_list:
        print("Expected lamda",diff_x.shape[0]*lam_test)
        suspected_indices = run_rave(diff_x,y_test_labels,lam =lam_test,ae_type=1,verbose=False)# 0.0022)
        #print("suspected_indices final",len(suspected_indices))
        print("*************************")
    #print("looking at preds only not diff")
    #suspected_indices = run_rave(xest_pred,y_test_labels,lam =lam_test)# 0.0022)
    #print("suspected_indices final",len(suspected_indices))
    #print("*************************")
    ##########
   
    return
def test_rvae_only(x_test,y_test_labels,lamda_list):
     """
        Run RVAE to detect spam in given data representation on a range of lamdas 
        We print Accuracy, Precsion, Recall & F1 score for each run
        Args:
            x_test: First matrix (3d numpy array) (#samples,#timestep,#features)
            y_test_labels: Label array with values (1: spam, 0:non-spam) (1d numpy array)
     """
    for lam_test in lamda_list:
         input_x = y_test.reshape(y_test.shape[0],y_test.shape[2])
         print("Now run rave only as is on test",input_x.shape,"for lamda",lam_test)
         suspected_indices = run_rave(input_x,y_test_labels,lam = lam_test,ae_type=1,verbose=True)
     
    return 
