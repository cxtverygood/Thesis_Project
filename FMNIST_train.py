import time
import psutil
import torch
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from nn_functions import Relu, Softmax, CrossEntropy
from FMNISTDataLoad import load_fashion_mnist_data
from trial import MotifBasedSparseNN

def get_cpu_info():
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    return cpu_count, cpu_count_logical

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(gpu_count):
            gpu_info.append({
                'name': torch.cuda.get_device_name(i),
                'cores': torch.cuda.get_device_properties(i).multi_processor_count,
                'total_memory': torch.cuda.get_device_properties(i).total_memory
            })
        return gpu_count, gpu_info
    else:
        return 0, []

def save_results_to_file(start_time, end_time, cpu_info, gpu_info, output_file, test_accuracy, epoch_info=None):
    with open(output_file, 'a') as f:
        if start_time and end_time:
            f.write(f"Program started at: {time.ctime(start_time)}\n")
            f.write(f"Program ended at: {time.ctime(end_time)}\n")
            f.write(f"Total execution time: {end_time - start_time} seconds\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        if cpu_info and gpu_info:
            f.write(f"\nCPU Information:\n")
            f.write(f"Physical cores: {cpu_info[0]}\n")
            f.write(f"Logical cores: {cpu_info[1]}\n")
            f.write(f"\nGPU Information:\n")
            if gpu_info[0] > 0:
                f.write(f"Number of GPUs: {gpu_info[0]}\n")
                for i, gpu in enumerate(gpu_info[1]):
                    f.write(f"GPU {i + 1}:\n")
                    f.write(f"  Name: {gpu['name']}\n")
                    f.write(f"  Cores: {gpu['cores']}\n")
                    f.write(f"  Total Memory: {gpu['total_memory']} bytes\n")
            else:
                f.write("No GPU available.\n")
        if epoch_info:
            for info in epoch_info:
                f.write(f"Epoch {info['epoch']}: Accuracy: {info['accuracy']:.4f}, Time: {info['time']:.2f} seconds\n")

def train_fashion_mnist(no_training_samples=5000, no_testing_samples=1000, motif_size=1, hidden_sizes=[300, 300], epochs=30, learning_rate=0.05, batch_size=40, zeta=0.1):
    # Data preparation
    X_train, y_train, X_test, y_test = load_fashion_mnist_data(no_training_samples, no_testing_samples)

    # Data Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot Encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

    # Set the parameters
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # Initialize and train the model
    nn = MotifBasedSparseNN(input_size, motif_size, hidden_sizes, output_size, init_network='uniform', epsilon=0.1, activation_fn=Relu(), loss_fn=CrossEntropy())

    best_accuracy = 0
    epoch_info = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        nn.train_epoch(X_train, y_train, learning_rate, batch_size, zeta)
        epoch_end_time = time.time()

        # 在测试集上进行预测
        _, A_list, _ = nn.forward(X_test)
        current_accuracy = nn.accuracy(y_test, A_list[-1])

        epoch_info.append({
            'epoch': epoch + 1,
            'accuracy': current_accuracy,
            'time': epoch_end_time - epoch_start_time
        })

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            print(f"New best accuracy at epoch {epoch + 1}: {best_accuracy:.4f}")

        save_results_to_file(None, None, None, None, 'epoch_results.txt', current_accuracy, [epoch_info[-1]])

    return best_accuracy

if __name__ == "__main__":
    start_time = time.time()

    test_accuracy = train_fashion_mnist()

    end_time = time.time()

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    save_results_to_file(start_time, end_time, cpu_info, gpu_info, 'program_results.txt', test_accuracy)
