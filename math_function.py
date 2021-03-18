import numpy as np
from numpy import linalg as la

def loss_function(m,H,w):
    # m is a matrix of points where rows are projective image points
    # H is the homography matrix
    # w is a matrix where rows are projective real points with only x y due to planar pattern
    sum = 0
    num_points = m.shape[0]
    for i in range(num_points):
        sum = sum + la.norm(m[i,:].T - H @ w[i,:].T,2)**2
    return sum

def testing_function():
    m = np.array([[1,2,1],[2,5,1]])
    w = np.array([[3,4,1],[5,6,1]])
    H = np.array([[1,2,3],[2,3,4],[3,4,5]])

    loss_value = loss_function(m,H,w)
    return loss_value

if __name__ == '__main__':
    print("Loss value: {}".format(testing_function()))