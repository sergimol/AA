from matplotlib import pyplot as plt
import numpy as np
import copy
import math

from yaml import load

from utils import load_data


#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """

    total_cost = 0
    m = x.shape[0]
    for i in range(m):
        total_cost += pow((w * x[i] + b) - y[i], 2)

    total_cost /= (2 * m)
    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """

    dj_dw = 0
    dj_db = 0
    m = x.shape[0]

    for i in range[m]:
        dj_dw += ((w * x[i] + b) - y[i]) * x[i]
        dj_db += (w * x[i] + b) - y[i]

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    m = len(x)

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y ,w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        #if i % math.ceil(num_iters, 10) == 0:
         #   print()

    return w, b, J_history

def plot_predictions(x, y, w, b):
    m = x.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * x[i] + b
    
    plt.plot(x, predicted, c="b")

    plt.scatter(x, y, marker='x', c ='r')

x, y = load_data()
w, b, J_History = gradient_descent(x, y, 0, 0, compute_cost, compute_gradient, 0.01, 1500)
plot_predictions(w, y, x, b)