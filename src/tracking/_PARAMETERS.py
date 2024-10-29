import numpy as np


# Create parameters

def generate_parameters(mode):

    if mode == 'Tracking':
        A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
        Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        R = np.array([[1, 0], [0, 1]])



        return A, B, Q, R





        