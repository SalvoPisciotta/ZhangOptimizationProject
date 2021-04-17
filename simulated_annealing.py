import numpy as np
import random
import math
import time
import ZhangOptimization as zo

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

def simulated_annealing(start, loss_function, m, w, low, upper, max_iter=1000,):
    "start is a np array, random_neighbour, acceptance and temperature are functions"
    # number of dimensions
    dim = start.shape[0]
    # generating starting point
    point = start
    value = loss_function(m, np.reshape(point,(3,3)), w)
    for step in range(max_iter):
        fraction = step / float(max_iter)
        t = temperature(fraction)
        neighbour = random_neighbour(point, low, upper, fraction)
        neighbour_value = loss_function(m, np.reshape(neighbour,(3,3)), w)
        if acceptance_probability(value, neighbour_value, t) > np.random.rand():
            point, value = neighbour, neighbour_value
    return (value, point)

def main():
    start = time.time()
    # Number of dimension
    DIM = 9
    # low and upper bounds
    LOW = 0
    UPPER = 1000
    # Set the name of the image file
    img = 'Chessboard.jpg'
    # Get m and w that represent respectively the image coordinates and the world coordinates already trasformed from R to P
    # It takes about 1 minute
    m , w = zo.process_corners(img)
    # Zhang optimization step (minimization of the distance from real coordinates in image plan and the ones found by the corner detector)
    # generating starting points
    np.random.seed(50)
    starting_point = random_start(LOW, UPPER, DIM)
    #for point in starting_points:
    #    print(point)
    # best_homography is a tuple
    best_homography = simulated_annealing(starting_point, zo.loss_function, m, w, LOW, UPPER)
    #Function that prints the points of the image and the projection error refering to the optimal H
    m = m[:,:2]
    w = np.reshape(best_homography[1],(3,3)) @ w.T
    w = w.T[:,:2]
    zo.print_correspondences(img,best_homography[1],best_homography[0],m,w)
    print("Time: {}".format(time.time() - start))

main()