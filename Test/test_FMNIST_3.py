import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from nn_functions import Relu, Softmax, CrossEntropy
from DataLoad import load_fashion_mnist_data

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

class MotifBasedSparseNN:
    def __init__(self, input_size, motif_size, hidden_sizes, output_size, init_network='uniform', epsilon=0.1, init_density=0.1, activation_fn=Relu(), loss_fn=CrossEntropy()):
        self.motif_size = motif_size
        self.epsilon = epsilon
        self.init_density = init_density
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn

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
            self.b.append(np.random.rand(hidden_size))
            prev_size = hidden_size

        self.W.append(create_network(self.epsilon, prev_size, output_size))
        self.b.append(np.random.rand(output_size))

    def forward(self, X):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        A = X
        Z_list = []
        A_list = [A]

        for i in range(len(self.hidden_sizes)):
            Z = np.zeros((A.shape[0], self.hidden_sizes[i]))
            for j in range(0, self.hidden_sizes[i], self.motif_size):
                motif_idx = j // self.motif_size
                W_submatrix = self.W[i][:, j:j+self.motif_size].toarray()

                Z[:, j:j+self.motif_size] = (A @ W_submatrix + self.b[i][j:j+self.motif_size])
            A = self.activation_fn.activation(Z)
            Z_list.append(Z)
            A_list.append(A)

        Z = np.dot(A, self.W[-1].toarray()) + self.b[-1]
        A = Softmax.activation(Z)

        Z_list.append(Z)
        A_list.append(A)

        return Z_list, A_list

    def backward(self, X, y_true, Z_list, A_list):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        m = y_true.shape[0]

        delta = self.loss_fn.delta(y_true, A_list[-1])
        dW = np.dot(A_list[-2].T, delta) / m
        db = np.sum(delta, axis=0) / m

        self.W[-1] = self.W[-1] - self.learning_rate * coo_matrix(dW)
        self.b[-1] = self.b[-1] - self.learning_rate * db

        for i in reversed(range(len(self.hidden_sizes))):
            delta = np.dot(delta, self.W[i + 1].toarray().T) * self.activation_fn.prime(Z_list[i])

            dW = np.zeros_like(self.W[i].toarray())
            for j in range(0, A_list[i].shape[1], self.motif_size):
                motif_idx = j // self.motif_size
                sub_delta = delta[:, j:j+self.motif_size]
                sub_A = A_list[i]

                dW[:, j:j+self.motif_size] = np.dot(sub_A.T, sub_delta) / m
                self.b[i][j:j+self.motif_size] -= self.learning_rate * np.sum(sub_delta, axis=0) / m

            self.W[i] = self.W[i] - self.learning_rate * coo_matrix(dW)

    def evolve(self):
        for i in range(len(self.W)):
            for j in range(len(self.W[i].data)):
                if random.random() < self.epsilon:
                    self.W[i].data[j] = 0

            self.W[i].data += (np.random.rand(*self.W[i].data.shape) < self.init_density).astype(float)

    def train(self, X, y_true, epochs, learning_rate):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            Z_list, A_list = self.forward(X)
            self.backward(X, y_true, Z_list, A_list)
            self.evolve()
            accuracy = self.accuracy(y_true, A_list[-1])
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    def accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true_labels, y_pred_labels)

# 数据准备
no_training_samples = 5000
no_testing_samples = 1000
X_train, y_train, X_test, y_test = load_fashion_mnist_data(no_training_samples, no_testing_samples)

# One-hot 编码
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

# 模型参数
input_size = X_train.shape[1]

motif_size = 4
hidden_sizes = [3000, 3000]
output_size = y_train.shape[1]

# 初始化和训练模型
nn = MotifBasedSparseNN(input_size, motif_size, hidden_sizes, output_size, init_network='uniform', epsilon=0.1, activation_fn=Relu(), loss_fn=CrossEntropy())
nn.train(X_train, y_train, epochs=3, learning_rate=0.05)

# 在测试集上进行预测
_, A_list = nn.forward(X_test)
test_accuracy = nn.accuracy(y_test, A_list[-1])
print(f"Test Accuracy: {test_accuracy:.4f}")
# 加一个batch size然后