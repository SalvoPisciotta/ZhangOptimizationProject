import numpy as np
import cv2

#Name of the file containing the image
dirname='Chessboard10'

#Load the image in greyscale color
img=cv2.imread('Chessboard10.jpg',cv2.IMREAD_GRAYSCALE)

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
pattern_points = np.concatenate([coordinates_3D, np.zeros([coordinates_3D.shape[0], 1], dtype=np.float32)], axis=-1)

print(coordinates_3D)
print(pattern_points)