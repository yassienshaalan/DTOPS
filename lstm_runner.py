import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.LSTM.LSTM_Model import *
from scipy import spatial
from statistics import  *


def run_lstm_model(X_train, y_train, l_s=5, l_p=1,verbose=False,model_num=-1):
    layers = [500, 500]
    lstm_model = LSTM_NETWORK(input_dim=X_train.shape[2], layers=layers, l_s=l_s, l_p=l_p)
    print("Input shapes",X_train.shape, y_train.shape)
    lstm_model.create_one_layer_model(input_dim=X_train.shape[2], layers=layers, l_s=l_s, l_p=l_p)
    #lstm_model.load_model(1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])

    # print("Reshaping",y_train.shape,y_test.shape)
    lstm_model.fit(X_train, y_train, epochs=100, validation_split=0.15, verbose=verbose,model_num=model_num)
    # y_hat = lstm_model.predict(X_test,y_test)
    return lstm_model
def load_lstm_trained_model(model_num,feat_dim):
    layers = [500, 500]
    lstm_model = LSTM_NETWORK(input_dim=feat_dim, layers=layers, l_s=5, l_p=1)
    model = lstm_model.load_model(model_num)
    return lstm_model
def check_aganist_test_and_plot(model,X_test,y_test):
    y_hat_2 = lstm_model.predict_all(X_test)
    #print("orig")
    dimensions = [1000,1010,1020,1030,1040]
    y_test_t = y_test.reshape(y_test.shape[0],y_test.shape[2])
    print("y_hat_2 shape", y_hat_2.shape, "y_test shape",y_test.shape)
    for i in range(5):
        train = []
        test = []
        pred = []
        for j in range(len(y_train)):
            train.append(y_train[j][dimensions[i]])
        for j in range(len(y_test_t)):
            test.append(y_test_t[j][dimensions[i]])
            pred.append(y_hat_2[j][dimensions[i]])
        print("train",len(train),"test",len(test),"pred",len(pred))
        #'''
        plt.plot(np.arange(0, len(train)), train, 'g', label="history")
        plt.plot(np.arange(len(train), len(train) + len(test)), test, marker='.', label="true")
        plt.plot(np.arange(len(train), len(train) + len(y_test)), pred, 'r', label="prediction")
        plt.ylabel('Value')
        plt.xlabel('Time Step')
        plt.legend()
        #plt.show()
        plt.title("Dimension"+str(i+1))
        plt.savefig("Dimension_"+str(i+1)+".png")
        #'''
        #break
    # measure similarities
    similarties = []
    for j in range(len(y_test)):
        similarties.append(1 - spatial.distance.cosine(y_test[j], y_hat_2[j]))
    print("Simlarties", len(similarties),"max", max(similarties),"min", min(similarties))
    print("mean", mean(similarties),"median", median(similarties))
    return
def predict_lstm(model,X_test,batches=1):
    if batches ==1:
        print("Predicting batches")
        y_hat = model.predict(X_test)
    else:
        print("Predicting All")
        y_hat = model.predict_all(X_test)
    return y_hat