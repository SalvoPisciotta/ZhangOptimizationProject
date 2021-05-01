# ZhangOptimizationProject
## Description
The purpose of the project is to minimize the reprojection error of the Zhang's method using two different optimization methods.
The Zhang's method is used to understand which are the camera parameters allowing to correlate image points with real points. Optimization methods used in the project to solve the first step of Zhang's methods are the followings:
* Nelder-Mead
* Particle Swarm Optimization

## Nelder-Mead method

<img align="left" heigth="350" width="350" src="https://rodolfoferro.files.wordpress.com/2017/02/gif1.gif">

The method uses the concept of a simplex, which is a special polytope of n + 1 vertices in n dimensions. Examples of simplices include a line segment on a line, a triangle on a plane, a tetrahedron in three-dimensional space and so forth.

The method approximates a local optimum of a problem with n variables when the objective function varies smoothly and is unimodal.

In this implementation has been used an adaptive Nelder-Mead method algorithm (ANMA) which allows to change coefficients according to the number of dimensions



## Particle Swarm Optimization (PSO) method
Different points are placed in the search space (particles) and each particle is updated considering the best position found by the swarm and found by itself.
In this implementation a local topology has been used to avoid a too fast convergence. Moreover a sort of correlation breaking has been implemented each 1000 iterations to improve the search and avoiding a stuck in a local minimum position.

<img align="center" src="https://upload.wikimedia.org/wikipedia/commons/e/ec/ParticleSwarmArrowsAnimation.gif">

## Requirements
You need to have a Python version < 3.7 and installed the following libraries:
* OpenCV
* Numpy
* Matplotlib

## Run
### Install
Clone the repository
```console
git clone https://github.com/SalvoPisciotta/ZhangOptimizationProject.git
cd ZhangOptimizationProject
```

### Run
From the project directory
```console
python main.py
```

## Group Members

|  Reg No.  |  Name     |  Surname  |     Email                              |    Username      |
| :-------: | :-------: | :-------: | :------------------------------------: | :--------------: |
|   985203  | Salvatore | Pisciotta | `salvatore.pisciotta2@studio.unibo.it` | [_SalvoPisciotta_](https://github.com/SalvoPisciotta) |
|  1005271  | Giuseppe  | Boezio    | `giuseppe.boezio@studio.unibo.it`      | [_giuseppeboezio_](https://github.com/giuseppeboezio) |
