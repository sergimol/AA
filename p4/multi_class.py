import copy
import numpy as np
import scipy.io as sio
#from public_tests import compute_gradient_reg_test
#from public_tests import compute_cost_reg_test

from utils import displayData

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

#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    m = len(X)
    sig = sigmoid(X @ w + b)
    total_cost = np.sum(-y * np.log(sig) - (1 - y) * np.log(1 - sig))
    w_sum = np.sum(pow(w, 2))    
    total_cost /= m
    w_sum = w_sum * lambda_ / (2*m)

    total_cost += w_sum
    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    dj_db = 0
    dj_dw = np.zeros(len(X[0]))
    m = len(X) 

    dj_db = np.sum(sigmoid(X @ w + b) - y)
    dj_dw = (sigmoid(X @ w + b) - y) @ X

    return dj_db / m, (dj_dw / m) + (np.dot(np.divide(lambda_, m), w))

#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y ,w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history += [cost_function(X, y, w, b, lambda_)]
            

    return w, b, J_history
    
#########################################################################
# one-vs-all
#
def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
     X : array_like
         The input dataset of shape (m x n). m is the number of
         data points, and n is the number of features. 

     y : array_like
         The data labels. A vector of shape (m, ).

     n_labels : int
         Number of possible labels.

     lambda_ : float
         The logistic regularization parameter.

     Returns
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """

    all_theta = np.zeros((n_labels, X.shape[1] + 1))   
    for i in range(n_labels):
        w = np.zeros(len(X[0]))
        b = 0
        y_aux = np.where(y == i, 1, 0)
        w_out, b_out, J_history = gradient_descent(X, y_aux, w, b, compute_cost_reg, compute_gradient_reg, 1, 1500, lambda_)
        all_theta[i, 0] = b_out
        all_theta[i, 1:] = w_out

    
    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """
    p = np.argmax(sigmoid(X @ all_theta[:, 1:].T + all_theta[:, 0]), 1)

    return p


#########################################################################
# NN
#
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
    a3 = np.argmax(a3.T, 1)                 
    return a3

data = sio.loadmat('p4/data/ex3data1.mat', squeeze_me=True)
X =  data['X']
y = data['y']

#Theta = oneVsAll(X, y, 10, 0.75)
#yP = predictOneVsAll(Theta, X)


weights = sio.loadmat('p4/data/ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
yP = predict(theta1, theta2, X)
percent_aux = np.where(y == yP, 1, 0)
percent = np.sum(percent_aux)
percent = percent * 100 / len(percent_aux)
print(percent)
#rand_indices = np.random.choice(X.shape[0], 100, replace=False)
#displayData(X[rand_indices, :])