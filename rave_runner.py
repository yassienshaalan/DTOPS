import tensorflow as tf
from models import RobustL21_X_Autoencoder as l21RDA
import numpy as np
from sklearn.metrics import (recall_score,f1_score,accuracy_score,precision_score)
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def calc_evaluations(y,good_indices):
    #contaminated_indices = []
    y_true = []#y.tolist()
    for i in range(len(y)):
        if y[i]==0:
            y_true.append(0)
        else:
            y_true.append(1)

    y_pred = [1]*len(y_true)
    for i in good_indices:
        y_pred[i]=0
    pred_array = np.array(y_pred)
    n_orig_spam = np.count_nonzero(y == 1)
    n_susp_spam = np.count_nonzero(pred_array == 1)
    #print("Num good",len(good_indices),"out of ",len(y))
    #print(commons)
    print("Originally spam",n_orig_spam,"#Suspected ",n_susp_spam,"/",len(y))
    #print(y[contaminated_indices])
    #commons = np.intersect1d(pred_array, y)
    #num_commons = len(commons) 
    #print("num commons automatic",num_commons)
    num_commons=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            num_commons+=1
    #print("num commons my way",num_commons,"acc",round(100*(num_commons/len(y_pred)),2))
    #print("len y_test",len(y_true),"pred",len(y_pred))
    #print("# ones in y_test",y_true.count(1))
    #print("# ones in y_pred", y_pred.count(1))
    #print("*******************************")
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)
    print('F1 score: %f' % f1)
    print("*******************************")
    return accuracy,precision,recall,f1

def evaluate_rave_performance(L,S,y):
    #print("Read","L",L.shape,"S",S.shape,"y",y.shape)
    #rows_with_zeros = np.any(S == 0, axis=1)
    tol=0.01
    #print("Saved S")
    S_t = np.zeros_like(S)
    #print("S_t",S_t.shape,S.shape)
    rows_non_zeros = np.any(abs(S-S_t)>tol, axis=1)#rows_with_zeros = np.any(abs(S-S_t)<=tol, axis=1)#np.any(S == 0, axis=1)
    good_indices = []
    suspected_indices = []
    for i in range(len(rows_non_zeros)):
        if rows_non_zeros[i]==False:#if rows_with_zeros[i]==True:
            good_indices.append(i)
        else:
            suspected_indices.append(i)
    #print("good_indices",len(good_indices))
    accuracy,precision,recall,f1=calc_evaluations(y,good_indices)
    return accuracy,precision,recall,f1,suspected_indices

def spam_detection_using_rave(X,y,lam,inner,outer,layers,verbose=False,shuffle=False):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            rael21 = l21RDA.RobustL21Autoencoder(sess=sess, input_shape=X.shape, lambda_=lam* X.shape[0],
                                                 layers_sizes=layers, type=type)
            l21L, l21S = rael21.fit(X=X, sess=sess, inner_iteration=inner, iteration=outer, batch_size=50,
                                    learning_rate=0.05, re_init=True, verbose=verbose)
            # l21R = rael21.getRecon(X=X_test, sess=sess)
            sess.close()
            return evaluate_rave_performance(l21L, l21S, y)


def run_rave(X,y,lam = 0.0030,ae_type = 1,verbose=False,shuffle=False):
    #lam = 0.0030  # smaller for bigger sets
    # lam = 0.0040 #bigger for smaller set
    # 0.00035
    layers = [X.shape[1], 500, 200]
    print("Lamda_index", lam)
    
    outer = 10  # looping between L and S
    inner = 100  # training the vae
    n_chunks = 1
    offset = int(len(X) / n_chunks)
    avg_acc = 0
    avg_prec = 0
    avg_rcl = 0
    avg_f1 = 0
    for i in range(0, len(X), offset):
        print(i, i + offset)
        Xtemp = X[i:i + offset]
        y_temp = y[i:i + offset]
        accuracy, precision, recall, f1,suspected_indices = spam_detection_using_rave(Xtemp, y_temp, lam, inner, outer,layers,verbose=verbose,shuffle=shuffle)
        avg_acc += accuracy
        avg_prec += precision
        avg_rcl += recall
        avg_f1 += f1
        print("*************************")
    if n_chunks>1:
        #print("Checking train set only")
        # spam_detection_using_rave(X_train,y_train,lam,inner,outer,layers)
        print("*************************")
        print("Average Accuracy:", str(avg_acc / n_chunks))
        print("Average Precision:", str(avg_prec / n_chunks))
        print("Average Recall:", str(avg_rcl / n_chunks))
        print("Average F1 score:", str(avg_f1 / n_chunks))

    return suspected_indices
