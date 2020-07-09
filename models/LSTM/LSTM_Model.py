from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.losses import mse, binary_crossentropy,cosine
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

class LSTM_NETWORK(object):
    def __init__(self, input_dim,layers,batch_size=32,l_s=5,l_p=1):
        """input_dim_list must include the original data dimension"""

        assert len(layers) >= 2
        self.l_s = l_s
        self.l_p = l_p
        self.batch_size = batch_size
        self.loss = 0#zero for mse, 1 for cosine similarity
        self.cbs = [History(),EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0003, verbose=0)]

        model = Sequential()
        model.add((LSTM(layers[0], input_shape=(l_s, input_dim),
                        return_sequences=True)))
                        #return_sequences=True)))
        model.add(Dropout(0.3))
        model.add(LSTM(layers[1], return_sequences=True))#return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Dense(self.l_p*input_dim))
        model.add(Activation("linear"))
        # model.add(Dense(activation='linear', units=y_train.shape[2]))
        if  self.loss == 0:
            model.compile(loss='mse', optimizer='adam')
        else:
            loss_fn = keras.losses.CosineSimilarity()
            model.compile(loss=loss_fn, optimizer='adam')
        # print("here is model summary")
        #print(model.summary())
        self.model = model
        return
    def create_one_layer_model(self,input_dim,layers,batch_size=32,l_s=5,l_p=1):
        assert len(layers) >= 2
        self.l_s = l_s
        self.l_p = l_p
        self.batch_size = batch_size
        self.cbs = [History(),EarlyStopping(monitor='val_loss', patience=15, min_delta=0.0003, verbose=0)]

        model = Sequential()
        model.add((LSTM(layers[0], input_shape=(None, input_dim))))
        model.add(Dropout(0.3))
        model.add(Dense(self.l_p*input_dim))
        model.add(Activation("linear"))
        # model.add(Dense(activation='linear', units=y_train.shape[2]))
        if  self.loss == 0:
            model.compile(loss='mse', optimizer='adam')
        else:
            loss_fn = keras.losses.CosineSimilarity()
            model.compile(loss=loss_fn, optimizer='adam')
        #import tensorflow as tf
        #model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer='adam')
        # print("here is model summary")
        #print(model.summary())
        #print("this is neww model")
        self.model = model
        return
    def fit(self, X,y, epochs=100,validation_split=0.15, verbose=False,model_num=-1):

        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=epochs,
                  validation_split=validation_split, verbose=verbose, callbacks=self.cbs)
        #print(history.history.keys())

        # "Accuracy"
        '''
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        '''

        # "Loss"
        '''
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        '''
        if model_num!=-1:
            self.model.save("LSTM_v"+str(model_num)+".h5")

        return

    def load_model(self,num):
        self.model=load_model(os.path.join("", "LSTM_v"+ str(num)+ ".h5"))
        return self.model

    def predict(self, X_test):
        '''
        Used trained LSTM model to predict test data arriving in batches
        Args:
                X_test (np array): numpy array of test inputs with dimensions [timesteps, l_s, input dimensions)

            Returns:
                y_hat (np array): predicted test values for each timestep in y_test
            '''
        print("Predicting by Patch")
        y_hat = []#np.array([[[]]])
        # print("y_hat intially",y_hat.shape)
        num_batches = int((X_test.shape[0] - self.l_s) / self.batch_size)
        print("number of batches",num_batches)
        if num_batches < 0:
            raise ValueError("l_s (%s) too large for stream with length %s." % (self.l_s, y_test.shape[0]))
        # simulate data arriving in batches
        for i in range(1, num_batches + 2):
            #print("Inside the loop")
            prior_idx = (i - 1) * self.batch_size
            idx = i * self.batch_size
            if i == num_batches + 1:
                idx = X_test.shape[0]  # remaining values won't necessarily equal batch size
            X_test_period = X_test[prior_idx:idx]
            #print("Predict for batch")
            #print("X_test_period",type(X_test_period),len(X_test_period))
            y_hat_period = self.model.predict(X_test_period)
            #print("y_hat_period out",y_hat_period.shape)
            #y_hat_period=np.array(y_hat_period)
            #print("y_hat_period after reshape",y_hat_period.shape)
            #print("y_hat now",y_hat_period.shape)
            if i ==1:
                y_hat =y_hat_period
            #y_hat_period=np.array(y_hat_period)
            #print("y_hat now",y_hat_period.shape)
            else:
                y_hat = np.append(y_hat, y_hat_period)
            #print("y_hat", y_hat.shape)
        print("Out of loop, final transformation")
        y_hat = y_hat.reshape(X_test.shape[0], X_test.shape[2])
        print("y_hat final", y_hat.shape)
        # np.save(os.path.join("data", anom['run_id'], "y_hat", anom["chan_id"] + ".npy"), np.array(y_hat))
        return y_hat
    def predict_all(self, X_test):
        '''
        Used trained LSTM model to predict test data arriving in batches
        Args:
                y_test (np array): numpy array of test outputs corresponding to true values to be predicted at end of each sequence
                X_test (np array): numpy array of test inputs with dimensions [timesteps, l_s, input dimensions)

            Returns:
                y_hat (np array): predicted test values for each timestep in y_test
            '''
        #print("Predicting All")
        y_hat = self.model.predict(X_test)
        #print("y_hat other",y_hat.shape)
        return  y_hat