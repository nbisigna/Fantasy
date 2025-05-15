import numpy as np

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # Z -= np.max(Z, axis=0) 
    return np.exp(Z) / np.sum(np.exp(Z))

def deriv_ReLU(Z):
    return Z > 0

def means_squared_loss(A2, Y):
    .5 * (A2 - Y) ** 2

def softmax_grad(s):
    print(s.shape)
    # input s is softmax value of the original input x. Its shape is (1,n) 
    # i.e.  s = np.array([0.3,0.7]),  x = np.array([0,1])

    # make the matrix whose size is n^2.
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1 - s[i])
            else: 
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m

def normalize(Y):
    return (Y - min(Y))/(max(Y) - min(Y))

def denormalize(A2, Y):
    return A2 * (max(Y) - min(Y)) + min(Y)

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


X = np.random.randn(1744, 4661)

print(X.shape)

W1 = np.random.randn(26, 1744)
# b1 = np.random.randn(26, 1)
W2 = np.random.randn(26, 1)
# b2 = np.random.randn(26, 1)
W3 = np.random.randn(1, 1)
# b3 = np.random.rand(1, 1)

Z1 = W1.dot(X)

print(Z1.shape)
A1 = ReLU(Z1)

Z2 = W2.dot(A1)
A2 = ReLU(Z2)
print(A2.shape)

Z3 = W3.dot(A2)
A3 = ReLU(Z3)
print(A3.shape)