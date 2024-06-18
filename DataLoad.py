# import numpy as np
# import pandas as pd

# def load_fashion_mnist_data(no_training_samples, no_testing_samples, random_seed=0):
#     np.random.seed(random_seed)

#     # 从 CSV 文件中读取数据
#     train_data = pd.read_csv("./FMNIST_Data/fashion-mnist_train.csv")
#     test_data = pd.read_csv("./FMNIST_Data/fashion-mnist_test.csv")

#     # 将标签和特征数据分开
#     x_train, y_train = train_data.iloc[:, 1:].values, train_data.iloc[:, 0].values
#     x_test, y_test = test_data.iloc[:, 1:].values, test_data.iloc[:, 0].values

#     # 随机抽样
#     index_train = np.random.choice(len(x_train), no_training_samples, replace=False)
#     index_test = np.random.choice(len(x_test), no_testing_samples, replace=False)
#     x_train, y_train = x_train[index_train], y_train[index_train]
#     x_test, y_test = x_test[index_test], y_test[index_test]

#     # 归一化至 0 到 1
#     x_train = x_train.astype('float64') / 255.
#     x_test = x_test.astype('float64') / 255.

#     return x_train, y_train, x_test, y_test
#Load data in this way
import numpy as np
import os
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def save_data_to_local(X_train, y_train, X_test, y_test, data_dir='fashion_mnist_data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    np.savez_compressed(os.path.join(data_dir, 'fashion_mnist_train.npz'), x=X_train, y=y_train)
    np.savez_compressed(os.path.join(data_dir, 'fashion_mnist_test.npz'), x=X_test, y=y_test)

def load_data_from_local(data_dir='fashion_mnist_data'):
    train_data = np.load(os.path.join(data_dir, 'fashion_mnist_train.npz'))
    test_data = np.load(os.path.join(data_dir, 'fashion_mnist_test.npz'))
    return train_data['x'], train_data['y'], test_data['x'], test_data['y']

def load_fashion_mnist_data(no_training_samples, no_testing_samples, random_seed=0, data_dir=None, standardize=False):
    np.random.seed(random_seed)

    if data_dir and os.path.exists(os.path.join(data_dir, 'fashion_mnist_train.npz')):
        X_train, y_train, X_test, y_test = load_data_from_local(data_dir)
    else:
        # 数据准备
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # 数据预处理
        X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # 归一化并展平
        X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

        # One-hot 编码
        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

        # 保存数据到本地
        if data_dir:
            save_data_to_local(X_train, y_train, X_test, y_test, data_dir)

    # 打乱训练集和测试集
    train_indices = np.random.permutation(X_train.shape[0])[:no_training_samples]
    test_indices = np.random.permutation(X_test.shape[0])[:no_testing_samples]

    X_train, y_train = X_train[train_indices], y_train[train_indices]
    X_test, y_test = X_test[test_indices], y_test[test_indices]

    # 数据标准化
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

# Example usage
if __name__ == "__main__":
    no_training_samples = 5000
    no_testing_samples = 1000
    data_dir = 'fashion_mnist_data'
    standardize = True

    X_train, y_train, X_test, y_test = load_fashion_mnist_data(
        no_training_samples, no_testing_samples, random_seed=42, data_dir=data_dir, standardize=standardize
    )

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")