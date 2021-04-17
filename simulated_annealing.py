import numpy as np
import random
import math

# Call it to generate the first point
def random_start(low, upper, num):
    """ Random point in the interval."""
    return np.random.uniform(low, upper, num)

def random_neighbour(point, low, upper, fraction=1):
    dim = point.shape[0]
    amplitudes = np.zeros(dim)
    delta = np.zeros(dim)
    for i in range(dim):
        amplitudes[i] = (upper - low) * fraction / 10
        delta[i] = (-amplitudes[i]/2.) + amplitudes[i] * np.random.rand()
    return point + delta

def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))

def acceptance_probability(value, new_value, temperature):
    if new_value < value:
        # new point is better than the previous one
        return 1
    else:
        p = np.exp(- (new_value - value) / temperature)
        # we accept non-improvement value with a certain probability
        return p

def simulated_annealing(start, loss_function, m, w, random_neighbour, low, upper, acceptance, temperature, max_iter=1000,):
    "start is a np array, random_neighbour, acceptance and temperature are functions"
    # number of dimensions
    dim = start.shape[0]
    # generating starting point
    point = start
    value = loss_function(m, np.reshape(point,(3,3)), w)
    for step in range(max_iter):
        fraction = step / float(max_iter)
        t = temperature(fraction)
        neighbour = random_neighbour(point, fraction)
        neighbour_value = loss_function(m, np.reshape(neighbour,(3,3)), w)
        if acceptance_probability(value, neighbour_value, t) > np.random.rand():
            point, value = neighbour, neighbour_value
    return (value, point)