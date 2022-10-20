import numpy as np

def load_data():
    data = np.loadtxt("data/houses.txt", delimiter=',', skiprows=1)
    X = data[:, :3]
    y = data[:, 3]

    return X, y