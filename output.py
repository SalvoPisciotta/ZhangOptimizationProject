from matplotlib import pyplot as plt
import cv2
import numpy as np

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

# Testing the function
shape = cv2.imread("Chessboard.jpg" ,cv2.IMREAD_GRAYSCALE).shape
print(shape)
print_correspondences("Chessboard.jpg", [1,2], 3, np.array([[650,1500], [650, 2500]]), np.array([[1200, 1600],[1100, 2300]]))