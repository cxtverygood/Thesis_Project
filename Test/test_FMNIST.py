import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist

import random  
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix, csr_matrix
from nn_functions import Relu, Sigmoid, Tanh, Softmax, CrossEntropy, MSE, NoActivation

class MotifBasedSparseNN:
    def __init__(self, input_size, motif_size, hidden_sizes, output_size, epsilon=0.1, init_density=0.1, activation_fn=Relu(), loss_fn=CrossEntropy()):
        self.motif_size = motif_size
        self.epsilon = epsilon  # 控制稀疏性
        self.init_density = init_density  # 初始密度
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        
        assert input_size % motif_size == 0
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # 这是一个包含每个隐藏层神经元数量的列表
        self.output_size = output_size
        
        # 初始化稀疏权重和偏置
        self.W = []
        self.b = []
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            assert hidden_size % motif_size == 0
            self.W.append(self.create_sparse_weights(prev_size, hidden_size))
            self.b.append(np.random.rand(hidden_size))
            prev_size = hidden_size
            
        self.W.append(self.create_sparse_weights(prev_size, output_size))
        self.b.append(np.random.rand(output_size))
    
    def create_sparse_weights(self, rows, cols):
        mask = np.random.rand(rows, cols) < self.init_density
        weights = np.random.rand(rows, cols) * mask
        return csr_matrix(weights)
    
    def forward(self, X):
        A = X
        Z_list = []
        A_list = [A]
        
        for i in range(len(self.hidden_sizes)):
            Z = np.zeros((A.shape[0], self.hidden_sizes[i]))
            for j in range(0, self.hidden_sizes[i], self.motif_size):
                motif_idx = j // self.motif_size
                W_submatrix = self.W[i][:, j:j+self.motif_size].toarray()  # 获取子矩阵
                
                # 调试信息
                print(f"Layer {i}, motif {motif_idx}")
                print(f"A[:, {j}:{j+self.motif_size}].shape: {A[:, j:j+self.motif_size].shape}")
                print(f"W_submatrix.shape: {W_submatrix.shape}")
                print(f"b[i][{j}:{j+self.motif_size}].shape: {self.b[i][j:j+self.motif_size].shape}")
                
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
                
                # 调试信息
                print(f"Layer {i}, motif {motif_idx}, sub_delta.shape: {sub_delta.shape}, sub_A.shape: {sub_A.shape}")
                
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
            accuracy = self.accuracy(y_true, A_list[-1])  # 计算准确率
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
    
    def accuracy(self, y_true, y_pred):
        """
        计算预测值的准确率
        :param y_true: 真实标签
        :param y_pred: 预测值
        :return: 准确率
        """
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true_labels, y_pred_labels)

# 数据准备
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # 归一化并展平
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# One-hot 编码
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

# 模型参数
input_size = 784
motif_size = 4
hidden_sizes = [3000, 3000]  # 确保隐藏层的大小是 motif_size 的倍数
output_size = 10

# 初始化和训练模型
nn = MotifBasedSparseNN(input_size, motif_size, hidden_sizes, output_size, activation_fn=Relu(), loss_fn=CrossEntropy())
nn.train(X_train, y_train, epochs=10, learning_rate=0.01)

# 在测试集上进行预测
_, A_list = nn.forward(X_test)
test_accuracy = nn.accuracy(y_test, A_list[-1])
print(f"Test Accuracy: {test_accuracy:.4f}")
