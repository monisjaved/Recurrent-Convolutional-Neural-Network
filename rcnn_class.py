"""
Recurrent Convolutional Neural Network Model For Text Classification
v0.1
Author: Monis Javed
Creation Date: 12th December 2015
Last Updated on: 24th December 2015
------------------------------------------------------------------------------------------------------

PREREQUISITE LIBRARIES:
'theano'          (for optimization mathematical expressions )
'numpy'           (to data conversion)
'sklearn'         (for generating confusion matrix)
'pickle'          (for loading and saving model)
'time'            (for computing time)

-----------
DESCRIPTION
-----------
This file contains a class named RCNN which is neural network model designed for
text classification. Traditional text classifiers often rely on many human-designed
features, such as dictionaries, knowledge bases, etc. So this model introduces a
recurrent convolutional neural network for text classification without human-designed
features. This model also learns the context of the word in which it is being used.

----------------
EXAMPLE USE-CASES
-----------------

t = TweetPreProcess()
x, y = t.create_set(filename="data/1700.csv", random=True)

vector_dict = theano.shared(np.asarray(t.word_vec).astype("float32"), name="vector_dict")

train_x_set = [x[i][:] for i in range(1000)]
train_y = [y[i][:] for i in range(1000)]

valid_x_set = [x[i][:] for i in range(1000, len(x))]
valid_y = [y[i][:] for i in range(1000, len(x))]

model = RCNN(vector_dict)
model.process_set(train_x_set,
                  train_y,
                  num_epochs=64,
                  verbose=True,
                  verbose_interval=1,
                  file_name="model_ada_small_check3",
                  valid_x=valid_x_set,
                  valid_y=valid_y)

-----------
PERFORMANCE
-----------
Even after optimization using theano, the computation takes time on training.It is
preferred to use GPU while training. If the computation is very slow you can disable
tuning of words embedding.

-----------
DATA USED
-----------
The pre-processing of the data is done by another class, i.e. TweetPreProcess. It

This model has been trained on a set of 1000 and tested on set of 532 with the maximum
testing accuracy of 82%

"""
import pickle
from collections import OrderedDict
import time

from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
from sklearn.metrics import confusion_matrix

from support_math import *

__author__ = 'Moonis Javed'

np.random.seed(130)
dtype = "float32"
theano.config.floatX = 'float32'


def numpy_floatX(data):
    """
    To convert data into data type supported by GPU
    """
    return np.asarray(data, dtype=theano.config.floatX)


class RCNN:
    def __init__(self, vector_dict=None, file_name=None):

        """
        Initializes all the parameters and build models

        Only one of the parameters should be mentioned.

        If vector_dict is mentioned model randomly initializes all the values
        else if file_name is mentioned then model will be overwritten loaded
        from that file.

        :type vector_dict: matrix
        :param vector_dict: matrix whose indexes defines the word embedding

        :type: string
        :param file_name: model will be loaded from the given file using load_model
        """
        self.srng = RandomStreams(seed=234)

        self.alpha = np.float32(0.01)  # Learning Rate
        self.reg_lambda = np.float32(0.0001)  # Regularization

        self.H = 100  # Number of context nodes
        self.V = 100  # Length of word embedding

        self.H1 = 2 * self.H + self.V  # 300 # 2H + Input Nodes, Input
        self.H2 = 150

        self.ClassSize = 5  # Number of Outputs {0, 1, 2, 3, 4}

        self.c0_l = theano.shared(0.01 * np.random.randn(self.H).astype(dtype),
                                  name="c0_l")  # Initial Convolution Weight
        self.c0_r = theano.shared(0.01 * np.random.randn(self.H).astype(dtype),
                                  name="c0_r")  # Initial Convolution Weight

        self.W_c_l = theano.shared(0.01 * np.random.randn(self.H + self.V, self.H).astype(dtype),
                                   name="W_c_l")  # Recurrent Context Weights (Combined)
        self.W_c_r = theano.shared(0.01 * np.random.randn(self.H + self.V, self.H).astype(dtype),
                                   name="W_c_r")  # Recurrent Context Weights (Combined)

        self.W_conv = theano.shared(0.01 * np.random.randn(self.H1, self.H2).astype(dtype), name="W_conv")
        # b_conv = theano.shared(0.01 * np.random.randn(ClassSize).astype(dtype), name="b_conv")

        self.W_output = theano.shared(0.01 * np.random.randn(self.H2, self.ClassSize).astype(dtype), name="W_output")
        self.b_output = theano.shared(0.01 * np.random.randn(self.ClassSize).astype(dtype), name="b_output")

        if vector_dict is not None:
            self.vector_dict = vector_dict
        else:
            self.vector_dict = theano.shared(np.asarray(np.random.randn(self.V, 1)).astype(dtype), name="vector_dict")

        if file_name is not None:
            self.load_model(file_name)
            print("Loading parameters from ", file_name)

        # Building Model
        print "Building Model. Please wait..."
        self.cost_function, self.prediction, self.f_shared_grad, self.f_update = self.build_model()
        print "Done"

    def convolution_layer(self, x, size):
        """
        Returns a theano variable defining outputs of convolution layer.

        :type x: Theano Variable
        :param x: Variable for a single set input variable

        :type size: Theano Variable
        :param size: Size of mask of the input variable

        :return: Theano Variable

        """
        x = x[:size]

        n = x.shape[0]

        '''
        # Randomize context
        l = self.srng.random_integers(low=1, high=n - 2)  # Length of string
        start = self.srng.random_integers(low=0, high=n - l)

        # x = x[start:(start + l)]
        n = x.shape[0]
        '''
        def propagate(x, c, W):
            p = T.concatenate((self.vector_dict[x], c))
            z = T.nnet.relu(p.dot(W))
            return z

        # Calculating left contexts
        f_left, _ = theano.scan(
            fn=propagate,
            sequences=x,
            outputs_info=self.c0_l,
            non_sequences=self.W_c_l,
            n_steps=n - 1
        )
        f_left = T.concatenate(([self.c0_l], f_left))

        # Calculating right contexts
        f_right, _ = theano.scan(
            fn=propagate,
            sequences=x,
            outputs_info=self.c0_r,
            n_steps=n - 1,
            non_sequences=self.W_c_r,
            go_backwards=True
        )
        f_right = T.concatenate((f_right, [self.c0_r]))

        # Combining and creating values for max-pooling layer
        def concat(x, left, right):
            return T.concatenate((x, left, right))

        y1, _ = theano.scan(
            fn=concat,
            sequences=(self.vector_dict[x], f_left, f_right),
            n_steps=n
        )

        y1 = y1.dot(self.W_conv)
        y1 = T.tanh(y1)

        return y1

    @staticmethod
    def max_pooling_layer(y1):
        """
        Returns theano variable returning input of output variable
        """
        y2 = y1.max(0)
        return y2

    def output_layer(self, y2):
        """
        Returns theano variable returning final probabilities of input
        at convolution layer
        """
        y3 = y2.dot(self.W_output)
        y3 += self.b_output
        final = T.nnet.softmax(y3)
        return final

    def get_prediction(self, x, size):
        """
        Uses all calculation functions an returns final probabilities
        for input parameters
        """
        y1 = self.convolution_layer(x, size)
        y2 = self.max_pooling_layer(y1)
        y3 = self.output_layer(y2)

        return y3

    def get_likelihood(self, x, y, size):
        """
        Uses get_prediction and calculates the cost of single input set.
        """
        y3 = self.get_prediction(x, size)

        cost = -(y * T.log(y3)).sum()

        return cost

    def get_cost(self, X, Y, X_sizes):
        """
        Calculates cost for each values in mini batch, also
        regularizes all the input parameters and then returns
        final cost function as theano variable

        """
        cost_fn, _ = theano.scan(
            fn=self.get_likelihood,
            sequences=[X, Y, X_sizes]
        )

        cost_fn = cost_fn.mean()
        cost_fn += self.reg_lambda * T.sqr(self.W_c_r).sum() / 2.
        cost_fn += self.reg_lambda * T.sqr(self.W_c_l).sum() / 2.
        cost_fn += self.reg_lambda * T.sqr(self.W_conv).sum() / 2.
        cost_fn += self.reg_lambda * T.sqr(self.W_output).sum() / 2.
        cost_fn += self.reg_lambda * T.sqr(self.b_output).sum() / 2.

        # Regularizing word embedding
        cost_fn += self.reg_lambda * T.sqr(self.vector_dict).sum() / 2

        return cost_fn

    def optimize(self, cost, X, Y, X_sizes):
        """
        Builds models for batch inputs and returns functions to update weights.

        :type cost: Theano Variable
        :param cost: Objective function to minimize

        :type X: Theano Matrix
        :param X: Masked matrix of all the training of a mini-batch

        :type Y: Theano Matrix
        :param Y: Matrix containing all expected outputs

        :type X_sizes: Theano Vector
        :param X_sizes: Vector containing actual sizes of training set

        :returns: (f_grad_shared, f_update)
            f_grad_shared: Theano Function
                :argument (X,Y, X_sizes) as defined for function
                :return cost of mini-batch, an pre-processes for parameter updating
            f_update: Theano Function
                :argument ()
                :return Update Parameters
        """
        params = OrderedDict({
            "c0_l": self.c0_l,
            "c0_r": self.c0_r,
            "W_c_l": self.W_c_l,
            "W_c_r": self.W_c_r,
            "W_conv": self.W_conv,
            "W_output": self.W_output,
            "b_output": self.b_output,
            "vector": self.vector_dict
        })

        gd = T.grad(cost, wrt=params.values())

        f_grad_shared, f_update = self.adadelta(params, gd, X, Y, X_sizes, cost)

        # Non shared variable cannot be update using updates so tunning of word embedding is not done

        return f_grad_shared, f_update

    def adadelta(self, tparams, grads, X, Y, X_sizes, cost):
        """
        An adaptive learning rate optimizer

        :type tparams: Theano SharedVariable
        :param tparams: Model Parameters

        :type grads: Theano variable
        :param grads: Gradients of cost w.r.t to parameters

        :type X: Theano variable
        :param X: Model inputs

        :type Y: Theano variable
        :param Y: Targets

        :type X_sizes: Theano variable
        :param X_sizes: Masking for model inputs

        :type cost: Theano variable
        :param cost: Objective function to minimize

        :returns: (f_grad_shared, f_update)
            f_grad_shared: Theano Function
                :argument (X,Y, X_sizes) as defined for function
                :return cost of mini-batch, an pre-processes for parameter updating
            f_update: Theano Function
                :argument ()
                :return Update Parameters

        """

        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]

        f_grad_shared = theano.function([X, Y, X_sizes], cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared')

        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

        f_update = theano.function([], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')

        return f_grad_shared, f_update

    def build_model(self):
        """
        Build models and returns a functions to control functions

        This function is automatically called at the initialization of object

        :return: (cost_evaluation, prediction, f_shared_grad, f_update)
            cost_evaluation: Theano Function
                :argument (X,Y, X_sizes)

                    :type X: Theano variable
                    :param X: Model inputs

                    :type Y: Theano variable
                    :param Y: Targets

                    :type X_sizes: Theano variable
                    :param X_sizes: Masking for model inputs

                :return cost of mini-batch
            prediction: Theano Function
                :argument (x, x_sizes)

                    :type x: list or vector
                    :param x: list of numbers defining word indexes in vector_dict

                    :type x_sizes: int
                    :param x_sizes: mask for input vector

                :return prediction of the input vector according to model
            f_grad_shared: Theano Function
                :argument (X,Y, X_sizes) as defined for function
                :return cost of mini-batch, an pre-processes for parameter updating
            f_update: Theano Function
                :argument ()
                :return Update Parameters
        """
        Y = T.matrix("Y", dtype=dtype)
        X = T.matrix("X", dtype="int32")
        X_sizes = T.vector("X_sizes", dtype="int32")

        x = T.vector("x", dtype="int32")
        x_size = T.scalar("x_size", dtype="int32")

        cost = self.get_cost(X, Y, X_sizes)

        pred = self.get_prediction(x, x_size)

        prediction = theano.function([x, x_size], pred)

        cost_eval = theano.function([X, Y, X_sizes], cost)

        f_shared_grad, f_update = self.optimize(cost, X, Y, X_sizes)

        return cost_eval, prediction, f_shared_grad, f_update

    def get_accuracy(self, pred, X, Y, X_sizes):
        """
        Computes accuracy for a mini-batch and returns accuracy in fraction

        :type pred: function(x, size)
        :param pred: Takes input as input array and mask and returns probability of each class

        :return: return accuracy of the the mini-batch in fraction
        """
        total = len(X)
        accurate = 0
        for x, y, size in zip(X, Y, X_sizes):

            h = pred(x, size)
            if np.argmax(h) == np.argmax(y):
                accurate += 1
        return float(accurate) / float(total)

    def save_model(self, file_name):
        """

        Saves the model as pickle.

        The model contains
            c0_l,
            c0_r,
            W_c_l,
            W_c_r,
            W_conv,
            W_output,
            b_output,
            vector_dict

        :type file_name: string
        :param file_name: filename to save model at
        """
        vars = (
            self.c0_l,
            self.c0_r,
            self.W_c_l,
            self.W_c_r,
            self.W_conv,
            self.W_output,
            self.b_output,
            self.vector_dict
        )
        vars_value = [c.get_value() for c in vars]
        with open("" + file_name, "wb") as fout:
            pickle.dump(vars_value, fout)

    def load_model(self, file_name):
        """
        Load and set model from a file

        :type file_name: string
        :param file_name: file from which model need to load with
        """
        vars = (
            self.c0_l,
            self.c0_r,
            self.W_c_l,
            self.W_c_r,
            self.W_conv,
            self.W_output,
            self.b_output,
            self.vector_dict
        )

        with open("" + file_name, "rb") as fin:
            vals = pickle.load(fin)

        [var.set_value(val) for var, val in zip(vars, vals)]

    def print_model(self):
        print "Word Vector Size: %d" % self.V
        print "Context Window Size: %d" % self.H
        print "Final Layer Size: %d" % self.H2
        print "Regularization Parameters: %f" % self.reg_lambda

    def print_confusion_matrix(self, X, Y, X_sizes):
        """
        Prints confusion matrix of the mini-batch provided as X, Y, X_sizes

        """
        total = len(X)
        actual = []
        pred_indexes = []
        for x, y, size in zip(X, Y, X_sizes):
            h = self.prediction(x, size)
            actual.append(np.argmax(y))
            pred_indexes.append(np.argmax(h))

        print "Confusion Matrix "
        print confusion_matrix(pred_indexes, actual)

    def process_set(self, data_x, data_y, num_epochs=300, verbose=True, verbose_interval=10, valid_x=None,
                    valid_y=None, file_name=None):
        """
        Trains the model with given data set for number for give number of times

        :rtype : object
        :type data_x: list of list (not a 2D list)
        :param data_x: data to train, don't pass the original list make a copy of it

        :type data_y: 2D array or vector defining exact outputs of network
        :param data_y: data to train, don't pass the original list make a copy of it

        :type num_epochs: integer
        :param num_epochs: number of epochs to train

        :type verbose: boolean
        :param verbose: Output at screen

        :type verbose_interval: int
        :param verbose_interval: Output will be shown in given interval of epochs

        :param valid_x: same as data_x
        :param valid_y: same as data_y

        :type file_name: string
        :param file_name: file to save best validation model

        :rtype: None
        """
        if valid_x is not None:
            valid_sizes = self.normalize_data(valid_x)

        v = False if valid_x is None else True

        save_file_name = False if file_name is None else file_name

        self.print_model()
        print("Train Size: %d" % len(data_x))
        if v: print("Validation Size: %d" % len(valid_x))

        max_accuracy = 0
        x_sizes = self.normalize_data(data_x)

        n = len(x_sizes)

        try:

            for i in range(num_epochs):

                start_time = time.time()

                mini_size = 64 * i / 128 + 3
                # mini_size = 16
                arr = get_minibatches_idx(n, mini_size, True)

                print "\n\nEpoch : %d" % i
                print "Mini Batch Size ", mini_size
                total_lik = 0
                for (k, order) in arr:
                    mini_train_x = [data_x[l] for l in order]
                    mini_train_y = [data_y[l] for l in order]
                    mini_x_sizes = [x_sizes[l] for l in order]

                    lik = self.f_shared_grad(mini_train_x, mini_train_y, mini_x_sizes)
                    self.f_update()

                    total_lik += lik

                    print "Mini Batch %d : %f : %f" % (k, lik, total_lik)

                # Printing
                if verbose:
                    if i % verbose_interval == 0:

                        print("Time taken for training : %s seconds" % (time.time() - start_time))
                        print "Train Likelihood: ", total_lik / len(arr)

                        train_acc = self.get_accuracy(self.prediction, data_x, data_y, x_sizes)

                        print "Train Accuracy: %3.3f %%" % (train_acc * 100.)

                        if v:
                            print "Validation Likelihood: ", self.cost_function(valid_x, valid_y, valid_sizes)

                            valid_accuracy = self.get_accuracy(self.prediction, valid_x, valid_y, valid_sizes)

                            print "Validation Accuracy: ", valid_accuracy
                            if save_file_name:
                                if valid_accuracy > max_accuracy:
                                    print "******************************************* Saving model with %3.3f %% accuracy" % (
                                        valid_accuracy * 100.0)
                                    self.save_model(save_file_name)
                                    max_accuracy = valid_accuracy
                                    self.print_confusion_matrix(data_x, data_y, x_sizes)
                                    self.print_confusion_matrix(valid_x, valid_y, valid_sizes)

        except KeyboardInterrupt:
            print "Training interrupted"
        except Exception:
            raise
        self.print_confusion_matrix(data_x, data_y, x_sizes)
        self.print_confusion_matrix(valid_x, valid_y, valid_sizes)

        return None

    @staticmethod
    def normalize_data(X):
        """
        This function adds padding to the X. Padding is added to
        convert data to matrix data type.

        Parameters
        ----------
        :type X: list of list
        :param X: Values to be added padding
        :rtype : list containing actual sizes of X
        """
        X_sizes = []
        size = max([len(case) for case in X])
        for case in X:
            l = len(case)
            X_sizes.append(l)
            case.extend([9999] * (size - l))
            # 9999 is used to make index error of anything doesn't work as planned
        return X_sizes

