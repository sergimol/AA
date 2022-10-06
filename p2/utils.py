import numpy as np

def load_data():
    data = np.loadtxt("data/houses.txt", delimiter=',', skiprows=1)
    X = data[:, :4]
    y = data[:, 4]

    return X, y