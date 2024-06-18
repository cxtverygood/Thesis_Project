import numpy as np
from numpy import ndarray
from numba import njit, prange
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from nn_functions import Relu, Softmax, CrossEntropy
from FMNISTDataLoad import load_fashion_mnist_data



@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask

def create_sparse_weights(epsilon, n_rows, n_cols):
    limit = np.sqrt(6. / float(n_rows))
    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)

    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights >= prob)
    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)

    return weights.tocsr()

def create_sparse_weights_normal_dist(epsilon, n_rows, n_cols):
    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)

    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights >= prob)
    weights[mask_weights >= prob] = np.random.randn(n_params) / 10

    return weights.tocsr()

def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    # TODO(Neil): not sure if we can asume uniqueness here
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return

class MotifBasedSparseNN:
    def __init__(self, input_size, motif_size, hidden_sizes, output_size, init_network='uniform', epsilon=20, zeta=0.1, activation_fn=Relu(), loss_fn=CrossEntropy(), init_density=0.1):
        self.motif_size = motif_size
        self.epsilon = epsilon
        self.zeta = zeta  # 添加 zeta 变量
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.learning_rate = None
        self.init_density = init_density
        self.dropout_rate = 0.

        if init_network == 'uniform':
            create_network = create_sparse_weights
        elif init_network == 'normal':
            create_network = create_sparse_weights_normal_dist
        else:
            raise ValueError("Unknown initialization method. Supports uniform and normal distribution")

        assert input_size % motif_size == 0

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.W = []
        self.b = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            assert hidden_size % motif_size == 0
            self.W.append(create_network(self.epsilon, prev_size, hidden_size))
            self.b.append(np.random.rand(hidden_size) * 0.01)  # Smaller initialization
            prev_size = hidden_size

        self.W.append(create_network(self.epsilon, prev_size, output_size))
        self.b.append(np.random.rand(output_size) * 0.01)  # Smaller initialization

    def forward(self, X, drop=False):
        A = X
        Z_list = []
        A_list = [A]
        masks = []

        for i in range(len(self.hidden_sizes)):
            Z = np.zeros((A.shape[0], self.hidden_sizes[i]))
            for j in range(0, self.hidden_sizes[i], self.motif_size):
                motif_idx = j // self.motif_size
                W_submatrix = self.W[i][:, j:j+self.motif_size].toarray()
                Z[:, j:j+self.motif_size] = (A @ W_submatrix + self.b[i][j:j+self.motif_size])
            A = self.activation_fn.activation(Z)
        
            if drop:
                A, mask = dropout(A, self.dropout_rate)
                masks.append(mask)
            else:
                masks.append(np.ones_like(A))

            Z_list.append(Z)
            A_list.append(A)

        Z = np.dot(A, self.W[-1].toarray()) + self.b[-1]
        A = Softmax.activation(Z)

        Z_list.append(Z)
        A_list.append(A)
        masks.append(np.ones_like(A))  # No dropout on the output layer

        return Z_list, A_list, masks

    def backward(self, X, y_true, Z_list, A_list, masks):
        m = y_true.shape[0]
        clip_value = 1.0  # Gradient clipping value

        # Calculate delta for the output layer
        delta = self.loss_fn.delta(y_true, A_list[-1])
        dW = np.dot(A_list[-2].T, delta) / m
        db = np.sum(delta, axis=0) / m

        # Apply gradient clipping
        np.clip(dW, -clip_value, clip_value, out=dW)
        np.clip(db, -clip_value, clip_value, out=db)

        self.W[-1] = self.W[-1] - self.learning_rate * coo_matrix(dW)
        self.b[-1] = self.b[-1] - self.learning_rate * db

        for i in reversed(range(len(self.hidden_sizes))):
            # Apply dropout mask during backpropagation
            delta = delta * masks[i+1]

            delta = np.dot(delta, self.W[i + 1].toarray().T) * self.activation_fn.prime(Z_list[i])

            dW = np.zeros_like(self.W[i].toarray())
            for j in range(0, A_list[i].shape[1], self.motif_size):
                motif_idx = j // self.motif_size
                sub_delta = delta[:, j:j+self.motif_size]
                sub_A = A_list[i]

                dW[:, j:j+self.motif_size] = np.dot(sub_A.T, sub_delta) / m
                self.b[i][j:j+self.motif_size] -= self.learning_rate * np.sum(sub_delta, axis=0) / m

            # Apply gradient clipping
            np.clip(dW, -clip_value, clip_value, out=dW)

            self.W[i] = self.W[i] - self.learning_rate * coo_matrix(dW)

    def get_threshold_interval(self, i=1, weights=None, threshold=None):
        if weights is None:
            weights = self.w[i]

        values = np.sort(weights.data)
        first_zero_pos = find_first_pos(values, 0)
        last_zero_pos = find_last_pos(values, 0)

        if not threshold:
            threshold = self.zeta

        largest_negative = values[int((1 - threshold) * first_zero_pos)]
        smallest_positive = values[
            int(min(values.shape[0] - 1, last_zero_pos + threshold * (values.shape[0] - last_zero_pos)))]

        return largest_negative, smallest_positive

    # counts the number of non-small (not close to zero) input connections per feature
    def get_core_input_connections(self, weights=None):
        if weights is None:
            weights = self.w[1]

        wcoo = weights.tocoo()
        vals_w = wcoo.data
        rows_w = wcoo.row
        cols_w = wcoo.col

        largest_negative, smallest_positive = self.get_threshold_interval(1, weights=weights)
        # remove the weights (W) closest to zero and modify PD as well
        pruned_indices = (vals_w > smallest_positive) | (vals_w < largest_negative)
        vals_w_new = vals_w[pruned_indices]
        rows_w_new = rows_w[pruned_indices]
        cols_w_new = cols_w[pruned_indices]

        return coo_matrix((vals_w_new, (rows_w_new, cols_w_new)), (self.dimensions[0], self.dimensions[1])).getnnz(
            axis=1)

    def vis_feature_selection(self, feature_selection):
        image_dim = (28, 28)
        f_data = np.reshape(feature_selection, image_dim)

        plt.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
        plt.title("Title")
        plt.show()

    def feature_selection(self, threshold=0.1, weights=None):
        """
        Selects the strongest features based on the number of strong connections of the input neuron

        :param threshold: the percentage of selected features TODO: Not really the percentage, more a mean dev. term
        :param weights: the weights to select from
        :return the strongest features
        """
        feature_strength = self.get_core_input_connections(weights=weights)

        absolute_threshold = (1 - threshold) * np.mean(feature_strength)

        feature_selection = feature_strength > absolute_threshold

        self.vis_feature_selection(feature_selection)

        return feature_selection

    def feature_selection_mean(self, sparsity=0.4, weights=None) -> ndarray:
        # TODO(Neil): explain why we choose only the first layer
        # the main reason is that this first layer will already have
        # most of the important information in it, given that everything
        # gets backpropageted

        if weights is None:
            weights = self.w[1]

        means = np.asarray(np.mean(np.abs(weights), axis=1)).flatten()
        means_sorted = np.sort(means)
        threshold_idx = int(means.size * sparsity)

        n = len(means)
        if threshold_idx == n:
            return np.ones(n, dtype=bool)

        means_threshold = means_sorted[threshold_idx]

        feature_selection = means >= means_threshold

        return feature_selection
        
    def evolve(self):
        for i in range(len(self.W)):
            # Remove a fraction of the weights
            mask = np.random.rand(len(self.W[i].data)) < self.zeta
            self.W[i].data[mask] = 0

            # Reinitialize some weights based on the density
            new_weights_mask = np.random.rand(len(self.W[i].data)) < self.init_density
            self.W[i].data += new_weights_mask.astype(float)

    def train(self, X, y_true, epochs, learning_rate, batch_size, drop=False, dropout_rate=0.5, zeta=0.1):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.zeta = zeta
        num_samples = X.shape[0]
        for epoch in range(epochs):
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch, y_batch = X[start:end], y_true[start:end]
                Z_list, A_list, masks = self.forward(X_batch, drop=drop)
                self.backward(X_batch, y_batch, Z_list, A_list, masks)
                self.evolve()
            accuracy = self.accuracy(y_true, self.forward(X)[1][-1])
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    def train_epoch(self, X, y_true, learning_rate, batch_size, drop=False, dropout_rate=0.5, zeta=0.1):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.zeta = zeta
        num_samples = X.shape[0]
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_batch, y_batch = X[start:end], y_true[start:end]
            Z_list, A_list, masks = self.forward(X_batch, drop=drop)
            self.backward(X_batch, y_batch, Z_list, A_list, masks)
            self.evolve()


    def accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true_labels, y_pred_labels)



