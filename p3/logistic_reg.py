import numpy as np
import copy
from utils import load_data, plot_decision_boundary

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
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    total_cost = 0
    m = X.shape[0]

    for i in range(len(X)):
      total_cost += (-y[i] * np.log(sigmoid(np.dot(w, X[i]) + b))) - (1 - y[i]) * np.log(1 - sigmoid(np.dot(w, X[i]) + b))
    
    total_cost /= m
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """
    dj_dw = 0
    dj_db = 0
    m = X.shape[0]

    for i in range(len(X)):
        dj_dw += (sigmoid(np.dot(w, X[i]) + b) - y[i]) * X[i]
        dj_db += (sigmoid(np.dot(w, X[i]) + b)) - y[i]


    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw


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
    total_cost = 0
    m = X.shape[0]

    for i in range(len(X)):
      total_cost += (-y[i] * np.log(sigmoid(np.dot(w, X[i]) + b))) - (1 - y[i]) * np.log(1 - sigmoid(np.dot(w, X[i]) + b))
    
    w_sum = 0
    for j in range(len(w)):
      w_sum += pow(w[j], 2)
    
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
    dj_dw = 0
    dj_db = 0
    m = X.shape[0]

    for i in range(len(X)):
        dj_dw += (sigmoid(np.dot(w, X[i]) + b) - y[i]) * X[i]
        dj_db += (sigmoid(np.dot(w, X[i]) + b)) - y[i]

    dj_dw /= m
    for j in range(len(w)):
      dj_dw[j] += (w[j] * lambda_ / m)
      
    dj_db /= m
    return dj_db, dj_dw


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
        dj_dw, dj_db = gradient_function(X, y ,w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w, b)
            J_history.append(cost)

    return w, b, J_history


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """

    m = len(X)
    p = np.zeros(m)
    for i in range(m):
      p[i] = sigmoid(np.dot(w, X[i]) + b)

    p = np.around(p)
    return p  

X, y = load_data()
w, b, J_history = gradient_descent(X, y, np.zeros(len(X[0])), 1, compute_cost_reg, compute_gradient_reg, 0.01, 10000, 0.01)
plot_decision_boundary(w, b, X, y)