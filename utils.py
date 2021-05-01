import numpy as np
from numpy import linalg as la
import cv2
from matplotlib import pyplot as plt
import time

def print_correspondences(image_dir, optima, value, first_list, second_list):
    '''
    image_dir: path of the image to show
    optima: optima point whose coordinates will be shown
    value: value of the optima using a function
    first_list: first list of points to show on the image, they must be a 2d numpy array
    second_list: second list of points to show on the image, they must be a 2d numpy array
    '''
    img = cv2.imread(image_dir ,cv2.IMREAD_GRAYSCALE)
    plt.title("Optima: {}\n Value: {}".format(optima, value))
    plt.imshow(img)
    # Showing first list of point
    plt.scatter(first_list[:,0],first_list[:,1], marker="o", color="red", label="Image point")
    # Showing second list of point
    plt.scatter(second_list[:,0],second_list[:,1], marker="o", color="green", label="Corresponding real point")
    plt.legend(loc='upper right')
    plt.show()


def process_corners(dir):
  #Load the image in greyscale color from the given directory path
  img=cv2.imread(dir,cv2.IMREAD_GRAYSCALE)

  #Set pattern size for openCv m=number corners in (columns,rows) coordinates
  #The  model proposed is always formed by 8x5 corners.
  pattern_size=(8,5)

  #Method that helps to find the corners into the image
  #And use a boolean variable found that is false in case of failure
  found , corners = cv2.findChessboardCorners(img, pattern_size)

  #If found is True then we use function cornerSubPix
  if found:
    #max_count=30 and criteria_eps_error=1 are the stop condition for define the corners
    term = (cv2.TERM_CRITERIA_EPS +  cv2.TERM_CRITERIA_COUNT , 30 ,1)
    #Redefine the corner positions
    cv2.cornerSubPix(img, corners, (5,5), (-1,-1), term)

  '''
  #Visualized found corners in the image
  vis=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  cv2.drawChessboardCorners(vis, pattern_size, corners, found)
  #plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
  #plt.show()
  '''

  #Setting square size in mm
  square_size=28
  #Build che coordinates i,j of coordinates for each point 
  #Method np.indices return an array containing 2D arrays
  indices=np.indices(pattern_size, dtype = np.float32)
  #Get real 3D coordinates
  indices *= square_size
  #Put them in a correct way with the transpose
  coordinates_3D=np.transpose(indices, [2,1,0])
  coordinates_3D=coordinates_3D.reshape(-1,2)
  #Concatenate the third axis z that is equal to 0
  coordinates_3D_points = np.concatenate([coordinates_3D, np.ones([coordinates_3D.shape[0], 1], dtype=np.float32)], axis=-1)
  corners_points = corners.reshape(-1,2)
  corners_image_points = np.concatenate([corners_points, np.ones([corners_points.shape[0], 1], dtype=np.float32)], axis=-1)

  #print(corners_image_points)
  #print(coordinates_3D_points)

  return (corners_image_points, coordinates_3D_points)

def loss_function(m,H,w):
    # m is a matrix of points where rows are projective image points
    # H is the homography matrix
    # w is a matrix where rows are projective real points with only x y due to planar pattern
    sum = 0
    num_points = m.shape[0]
    for i in range(num_points):
        sum = sum + la.norm(m[i,:].T - H @ w[i,:].T,2)**2
    return sum