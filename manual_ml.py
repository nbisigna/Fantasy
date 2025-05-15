import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./draft_prediction_denormalized.csv')

# data.head()

data = np.array(data)
m, n = data.shape
# 5661 rows x 1744 columns
# np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[n-1]
X_dev = data_dev[0:n-1]
# X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[n-1]
X_train = data_train[0:n-1]
# X_train = X_train / 255.
m = m - 1000

X_train = X_train.T

def init_params():
    W1 = np.random.randn(1, 1743) * np.sqrt(1/1743)
    b1 = np.random.randn(1, 1743) * np.sqrt(1/1743)
    W2 = np.random.randn(1, 1) * np.sqrt(1/1)
    b2 = np.random.randn(1, 1) * np.sqrt(1/1)
    # W3 = np.random.randn()
    # b3 = np.random.randn()
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Z -= np.max(Z, axis=0) 
    return np.exp(Z) / np.sum(np.exp(Z))

def foward_prop(W1, b1, W2, b2, X, Y):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    return Z1, A1, Z2, A2

# def one_hot(Y):
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#     one_hot_Y[np.arange(Y.size), Y] = 1
#     return one_hot_Y.T

def deriv_ReLU(Z):
    return Z > 0

def means_squared_loss(A2, Y):
    return np.sqrt(.5 * (A2 - Y) ** 2)

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    dZ2 = means_squared_loss(A2, Y)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_accuracy(A2, Y):
    return np.mean(A2)/ Y * 100

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = foward_prop(W1, b1, W2, b2, X[i], Y[i])
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X[i], Y[i])
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f'Iteration: {i}')
            print(A2, Y[i])
            print(f'Accuracy {get_accuracy(A2, Y[i])}')
            print('W1', W1)
            print('b1', b1)
            print('W2', W2)
            print('b2', b2)
    return W1, b1, W2, b2

# Y_train = normalize(Y_train)
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 3, 5)

def make_predictions(X, Y, W1, b1, W2, b2):
    _, _, _, A2 = foward_prop(W1, b1, W2, b2, X, Y)
    return np.sum(A2)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[index]
    prediction = make_predictions(X_train[index], Y_train, W1, b1, W2, b2)
    label = Y_train[index]
    print(f'Prediction: {prediction}')
    print(f'Label {label}')
    print(current_image)
    # current_image = current_image.reshape((28, 28)) * 225
    # plt.gray()
    # plt.imshow(current_image, interpolation='nearest')
    # plt.show()

test_prediction(2, W1, b1, W2, b2)


# Y_dev = normalize(Y_dev)

# dev_predictions = make_predictions(X_dev, Y_dev, W1, b1, W2, b2)
# get_accuracy(dev_predictions, Y_dev)