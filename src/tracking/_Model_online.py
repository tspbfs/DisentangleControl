import numpy as np
import math
import control
import random


# Define trackings

def tracking_coordinates(theta):
    R, r, d = 6.0/2, 1.0/2, 3.0/2
    
    dem = 240 / (4 * np.pi) *2
    x = (R - r) * np.cos(theta/dem) + d * np.cos((R - r) * theta / r/dem)
    y = (R - r) * np.sin(theta/dem) - d * np.sin((R - r) * theta / r/dem)
    return x,y




def generate_w(mode, A, T):

    w = np.zeros((T, np.shape(A)[0]))

    if mode == 'Tracking':

        for t in range(T):
            y_1, y_2 = tracking_coordinates(t)
            y_3, y_4 = tracking_coordinates(t + 1)

            # Ground-true 
            w[t] = np.matmul(A, np.array([y_1, y_2, 0, 0])) - np.array([y_3, y_4, 0, 0])


    return w















