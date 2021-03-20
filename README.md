# ZhangOptimizationProject
The purpose of the project is to minimize the reprojection error of the Zhang's method using the Nelder Mead method.

## First Step
Given an image with a planar pattern (e.g. a chessboard), corner points are extracted and knowing the real dimension of each square of the chessboard is possible to compute real world points.

<img align="center" src="https://automaticaddison.com/wp-content/uploads/2020/12/7_chessboard_input1_drawn_corners.jpg" >
                                                                                                        

## Second Step
Nelder Mead method is applied to minimize the reprojection error i.e. the sum of the square distances between the image points and the image points obtained using the homography matrix (this matrix is used to realize a change of coordinates between image reference system and world reference system).

<img align="center" src="https://rodolfoferro.files.wordpress.com/2017/02/gif1.gif" >

