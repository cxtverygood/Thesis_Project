import numpy as np
from sklearn.metrics import log_loss

# Contains different activation functions and losses to train your set mlp model


class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class Tanh:
    @staticmethod
    def activation(z):
        return np.tanh(z)

    @staticmethod
    def prime(z):
        return 1 - Tanh.activation(z) ** 2


class Sigmoid:

    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class Softmax:
    @staticmethod
    def activation(z):
        """
        https://stackoverflow.com/questions/34968722/softmax-function-python
        Numerically stable version
        """
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

# class Softmax:
#     @staticmethod
#     def activation(z):
#         """
#         Computes the softmax activation function in a numerically stable way.
        
#         Parameters:
#         z (numpy.ndarray): Input array of shape (n_samples, n_features).
        
#         Returns:
#         numpy.ndarray: Output array of the same shape as input, representing the softmax probabilities.
#         """
#         # Subtract the maximum value from each row for numerical stability
#         z_shifted = z - np.max(z, axis=1, keepdims=True)
#         exps = np.exp(z_shifted)
#         return exps / np.sum(exps, axis=1, keepdims=True)

class CrossEntropy:
    """
    Used with Softmax (multi-class) or Sigmoid (binary classification) activation in final layer
    """

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return y_pred - y_true

    @staticmethod
    def loss(y_true, y):
        return log_loss(y_true, y)


class MSE:
    def __init__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class NoActivation:
    """
    This is a plugin function for no activation.

    f(x) = x * 1
    """

    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        return z

    @staticmethod
    def prime(z):
        """
        The prime of z * 1 = 1
        :param z: (array)
        :return: z': (array)
        """
        return np.ones_like(z)
