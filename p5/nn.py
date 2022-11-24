import numpy as np
import scipy.io as sio
from utils import checkNNGradients
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1 / (1 + np.exp(np.negative(z)))

    return g

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    a1 = np.c_[np.ones(len(X)), X]          
    z2 = np.dot(theta1, a1.T)           
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(len(a2[0])), a2.T]   
    z3 = np.dot(theta2, a2.T)
    a3 = sigmoid(z3)                    
    a3 = a3.T              
    return a3, a2, a1

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    a3, a2, a1 = predict(theta1, theta2, X)
    m = len(X)
    
    c = np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
    l = np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2)

    J = -c/m + (lambda_/(2*m)*l)
    return J



def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """
    grad1 = np.zeros([len(theta1), len(theta1[0])])
    grad2 = np.zeros([len(theta2), len(theta2[0])])
    m = len(X)

    for i in range(m):
        a3, a2, a1 = predict(theta1, theta2, [X[i]])

        sigma3 = a3 - y[i]
        gPrima = a2 * (1 - a2)
        sigma2 = np.dot(sigma3, theta2) * gPrima

        # sigma es sigma2 sin la primera columna
        sigma = sigma2[:, 1:]

        grad1 += np.dot(sigma.T, a1)
        grad2 += np.dot(sigma3.T, a2)

    grad1[:,1:] += lambda_*theta1[:,1:]
    grad2[:,1:] += lambda_*theta2[:,1:]

    J = cost(theta1, theta2, X, y, lambda_)
    return (J, grad1 / m, grad2 / m)

data = sio.loadmat('p5/data/ex3data1.mat', squeeze_me=True)
X = data['X']
y = data['y']
y_hot = np.zeros([len(y), 10])
for i in range(len(y)):
    y_hot[i][y[i]] = 1

weights = sio.loadmat('p5/data/ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
checkNNGradients(backprop, 1)
#c = cost(theta1, theta2, X, y_hot, 1)
#print(c)