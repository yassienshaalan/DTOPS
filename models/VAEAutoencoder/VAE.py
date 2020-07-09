import tensorflow as tf
import numpy as np
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy,cosine
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.losses
from keras.callbacks import History, EarlyStopping, Callback


import os

def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i, min(l, i + n))


class VAE_Autoencoder(object):
    def __init__(self, sess, input_shape,input_dim_list=[],latent_dim=3):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.dim_list=input_dim_list
        # VAE model = encoder + decoder
        #print("input_shape",input_shape)
        original_dim = input_shape[1]
        # build encoder model
        #*********************************************************************
        inputs = Input(shape=(original_dim,), name='encoder_input')
        x=inputs
        for i in range(1,len(self.dim_list)):
            x = Dense(self.dim_list[i], activation='relu')(x)
        #print("xy",x)
        self.z_mean = Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        def sampling(args):
            """Reparameterization trick by sampling from an isotropic unit Gaussian.
            # Arguments
                args (tensor): mean and log of variance of Q(z|X)
            # Returns
                z (tensor): sampled latent vector
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        #print("Encoder",self.encoder.summary())
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        # *********************************************************************
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = latent_inputs
        for i in range(1, len(self.dim_list)):
            x = Dense(input_dim_list[i], activation='relu')(x)
        outputs = Dense(input_dim_list[0], activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        #print("Decoder",self.decoder.summary())
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        loss_type = 0
        if loss_type ==0:
            print("The chosen loss mse")
            self.reconstruction_loss = mse(inputs, outputs)
        elif loss_type ==1:
            print("The chosen loss binary crossentropy")
            self.reconstruction_loss = binary_crossentropy(inputs,outputs)
        elif loss_type == 2:
            print("The chosen loss cosine")
            self.reconstruction_loss = cosine(inputs, outputs)

        #add one for cosine as well

        self.reconstruction_loss *= original_dim
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(self.reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam', loss = None)
        #   print(self.vae.summary())
        #plot_model(self.vae, to_file='vae_mlp.png', show_shapes=True)

        # reparameterization trick
        # instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # z = z_mean + sqrt(var) * epsilon

    def fit(self, X, sess, learning_rate=0.15,
            iteration=100, batch_size=30, init=False, verbose=False,y=[],shuffle=False):
        assert X.shape[1] == self.dim_list[0]
        if shuffle == True:
            X_train, X_valid = train_test_split(X,test_size = 0.2, random_state = 42)
        else:
            num_train = int(len(X) * 0.8)
            X_train = X[:num_train]
            X_valid = X[num_train:]
        '''
        else:
            print("Y not None")
            X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
            print("Original train", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        '''
        #X_train_new, X_valid = train_test_split(X_train, test_size=0.2, random_state=42)
        #print("Original train", X_train_new.shape, X_valid.shape)
        cbs = [History(), EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0003, verbose=0)]
        self.vae.fit(X_train,epochs=iteration,batch_size=batch_size,validation_data=(X_valid, None),verbose=verbose,callbacks=cbs)
        models = (self.encoder, self.decoder)
        #self.plot_results(X_test,y_test,model_name="vae_mlp")
    def transform(self, X, sess):
        return self.hidden.eval(session=sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess,batch_size=50):
        z_mean, _, _ = self.encoder.predict(X, batch_size=batch_size)
        #print(type(z_mean),len(z_mean))

        x_decoded = self.decoder.predict(z_mean)
        #print("Xdecoded",x_decoded.shape)
        return x_decoded

        #return self.recon.eval(session=sess, feed_dict={self.input_x: X})


    def plot_results(self,
                     x_test, y_test,
                     batch_size=128,
                     model_name="vae_mnist"):
        """Plots labels and MNIST digits as a function of the 2D latent vector
        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        os.makedirs(model_name, exist_ok=True)

        filename = os.path.join(model_name, "vae_mean.png")
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(x_test,batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename)
        plt.show()

        filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()

##################### test a machine with different data size#####################
def test():
    start_time = time.time()
    with tf.Session() as sess:

        original_dim=784
        vae = VAE_Autoencoder(sess=sess,input_shape=x.shape,input_dim_list=[784, 625])

        error = vae.fit(x[:1000],y=y[:1000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=True)
        vae.getRecon(x[:1000], sess=sess)
    print("size 1000 Runing time:" + str(time.time() - start_time) + " s")

    '''
    start_time = time.time()
    with tf.Session() as sess:
        ae = VAE_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:10000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 10,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = VAE_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:20000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 20,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = VAE_Autoencoder(sess=sess, input_dim_list=[784, 625, 400, 225, 100])
        error = ae.fit(x[:50000], sess=sess, learning_rate=0.01, batch_size=500, iteration=1000, verbose=False)

    print("size 50,000 Runing time:" + str(time.time() - start_time) + " s")

    '''
if __name__ == "__main__":
    import time
    import os

    os.chdir("../../")
    x = np.load(r"./data/data.npk")
    y = np.load(r"./data/y.npk")
    start_time = time.time()
    '''
    with tf.Session() as sess:
        ae = VAE_Autoencoder(sess=sess, input_dim_list=[784, 225, 100])
        error = ae.fit(x, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
        R = ae.getRecon(x, sess=sess)
        print("size 100 Runing time:" + str(time.time() - start_time) + " s")
        error = ae.fit(R, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
        '''
    test()
