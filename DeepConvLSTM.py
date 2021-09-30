import tensorflow
import printer as printer
from numba import cuda
from sklearn.model_selection import train_test_split

# author: Alexander Hoelzemann - alexander.hoelzemann@uni-siegen.de

# Clear allocated memory on GPU device
try:
    device = cuda.get_current_device()
    device.reset()
except:
    pass

# Following three lines can be executed in case your gpu has memory issues during training
config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)


###############################################################
##            Class for customized callbacks                 ##
##          Can be called each epoch, batch etc.             ##
###############################################################

class CustomCallbacks(tensorflow.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass


class CustomMetrics():
    # copied from unnir's post https://github.com/keras-team/keras/issues/5400

    def recall_m(y_true, y_pred):
        true_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tensorflow.keras.backend.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tensorflow.keras.backend.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = CustomMetrics.precision_m(y_true, y_pred)
        recall = CustomMetrics.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tensorflow.keras.backend.epsilon()))



class DeepConvLSTM:

    ###############################################################
    ##                    Class-Initialization                   ##
    ##                   and Network definition                  ##
    ###############################################################

    def __init__(self, Xs, Ys, X_test=None, Y_test=None, modelname='DeepConvLSTM', verbose=None, epochs=None,
                 batch_size=None, validation_split=None, learning_rate=0.01, regularization_rate=0.01,
                 regularization=tensorflow.keras.regularizers.l2,
                 optimizer=tensorflow.keras.optimizers.Adam, use_batch_norm=False, rnn_size=128,
                 num_rnn_layers=2, dropout_rate=0.5, filter_size=[64, 128, 256], samples_per_convolution_step=5,
                 samples_per_pooling_step=0, weight_init='lecun_uniform'):

        '''
        :param Xs: x_train: ndarray
        :param Ys: y_train: ndarray
        :param X_test: x_test: ndarray
        :param Y_test: y:test: ndarray
        :param modelname: string
        :param verbose: int between 0 and 2
        :param epochs: number of training epochs: int
        :param batch_size: int
        :param validation_split: variable that defines the percentage of training data used for validation: float between 0.01 and 0.99
        :param window_length: window length as number of samples: int
        :param learning_rate: float
        :param regularization: desired regularization as object from tensorflow.keras.regularizers
        :param random_seed: int
        :param optimizer: desired optimizers as object from tensorflow.keras.regularizers
        :param use_batch_norm: set True if batch normalization should be used : boolean
        :param rnn_size: int
        :param num_rnn_layers: int
        :param dropout_rate: float between 0.1 and 0.9
        :param filter_size: list of filter sizes for every convolutational layer - every element is an int
        :param samples_per_convolution_step: int
        :param samples_per_pooling_step: int - if you would like to use MaxPooling, change to an int > 0
        :param weight_init: string
        '''

        self.X_train = Xs
        self.y_train = Ys
        self.X_test = X_test
        self.y_test = Y_test
        if validation_split > 0.0:
            self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train,
                                                                                                self.y_train,
                                                                                                test_size=validation_split,
                                                                                                shuffle=False)
        else:
            pass

        self.verbose = verbose
        self.epochs = epochs
        self.validation_split = validation_split

        # Settings for HyperParameters
        self.batch_depth = self.X_train.shape[0]
        self.batch_length = self.X_train.shape[1]
        self.n_channels = self.X_train.shape[2]
        self.n_classes = self.y_train.shape[1]
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.optimizer = optimizer(lr=learning_rate)
        self.regularization = regularization(l=regularization_rate)
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_rate = dropout_rate
        self.filter_size = filter_size
        self.num_cnn_layers = len(self.filter_size)
        self.modelname = modelname
        self.use_batch_norm = use_batch_norm
        self.kernel_size = (samples_per_convolution_step, self.n_channels)
        self.pool_size = (samples_per_pooling_step, 1)

    def init_network(self):

        inputs = tensorflow.keras.Input(shape=(self.batch_length, self.n_channels))
        if self.use_batch_norm:
            x = tensorflow.keras.layers.BatchNormalization(name='input/batch_norm',
                                                           input_shape=(self.batch_length, self.n_channels))(inputs)
        x = tensorflow.keras.layers.Reshape(name='reshape_to_3d', target_shape=(self.batch_length, self.n_channels, 1))(
            inputs)
        for cnn_layer in range(0, self.num_cnn_layers):
            x = tensorflow.keras.layers.Convolution2D(name='conv2d_' + str(cnn_layer),
                                                      filters=self.filter_size[cnn_layer], kernel_size=self.kernel_size,
                                                      padding='same',
                                                      kernel_regularizer=self.regularization,
                                                      bias_regularizer=self.regularization,
                                                      kernel_initializer=self.weight_init)(x)

            if self.use_batch_norm:
                x = tensorflow.keras.layers.BatchNormalization()(x)
            x = tensorflow.keras.layers.Activation('relu')(x)
        if self.pool_size[0] > 0:
            x = tensorflow.keras.layers.MaxPooling2D(self.pool_size[0])(x)
        x = tensorflow.keras.layers.Reshape(name="reshape_to_1d",
                                            target_shape=(x.shape[1], self.filter_size[-1] * self.n_channels))(x)
        for rnn_layer in range(0, self.num_rnn_layers):
            if rnn_layer == 0:
                x = tensorflow.keras.layers.LSTM(self.rnn_size,
                                                 return_sequences=True)(x)
            else:
                x = tensorflow.keras.layers.LSTM(self.rnn_size, return_sequences=False)(x)

        x = tensorflow.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tensorflow.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        self.neural_network = tensorflow.keras.Model(inputs, outputs, name=self.modelname)
        self.neural_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer,
                                    metrics=['accuracy', CustomMetrics.f1_m, CustomMetrics.precision_m,
                                             CustomMetrics.recall_m])

        printer.write('New network intialized', 'blue')

    def fit(self):
        if self.validation_split > 0.0:
            return self.neural_network.fit(self.X_train, self.y_train, epochs=self.epochs,
                                           batch_size=self.batch_size,
                                           verbose=self.verbose,
                                           validation_data=(self.X_validation, self.y_validation),
                                           callbacks=[CustomCallbacks()])
        else:
            return self.neural_network.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                           verbose=self.verbose, callbacks=[CustomCallbacks()])

    ###############################################################
    ##            Methods to evaluate the model                  ##
    ###############################################################

    def evaluate_model(self):
        self.evaluation_results = {}
        self.evaluation_results['loss'], self.evaluation_results['accuracy'], \
        self.evaluation_results['f1_score'] = self.neural_network.evaluate(self.X_train, self.y_train,
                                                                           verbose=self.verbose)

    ###############################################################
    ##      execute the experiment: and train the neural network ##
    ###############################################################

    def execute_experiment(self):

        self.init_network()
        history = self.fit()
        predictions = self.neural_network.predict(x=self.X_test)

        return history, predictions


# Example
# my_neural_network = DeepConvLSTM(Xs=X_train, Ys=y_train, X_test=X_test, Y_test=y_test,
#                                 verbose=2, epochs=30, batch_size=64, window_length=50, validation_split=0.1)
# history, predictions = my_neural_network.execute_experiment()
#
# Compare predictions to self.y_test