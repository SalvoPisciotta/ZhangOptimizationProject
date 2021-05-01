# ZhangOptimizationProject

This is a project for the second module of Combinatorial Decision Making and Optimization of the degree course of Artificial Intelligence at Alma Mater Studiorum (Bologna)

## Description
The purpose of the project is to minimize the reprojection error of the Zhang's method using two different optimization methods.

Zhang's method is used to understand which are the camera parameters which allow to correlate image points with real points. Optimization methods used in the project to solve the first step of Zhang's method are the followings:
* Nelder-Mead
* Particle Swarm Optimization

## Nelder-Mead method

<img align="left" heigth="350" width="350" src="https://rodolfoferro.files.wordpress.com/2017/02/gif1.gif">

The method uses the concept of a simplex, which is a special polytope of n + 1 vertices in n dimensions. Examples of simplices include a line segment on a line, a triangle on a plane, a tetrahedron in three-dimensional space and so forth.

The method approximates a local optimum of a problem with n variables when the objective function varies smoothly and is unimodal.

The downhill simplex method now takes a series of steps, most steps just moving the point of the simplex where the function is largest (“highest point”) through the opposite face of the simplex to a lower point. These steps are called reflections, and they are constructed to conserve the volume of the simplex (and hence maintain its nondegeneracy). When it can do so, the method expands the simplex in one or another direction to take larger steps. When it reaches a “valley floor”, the method contracts itself in the transverse direction and tries to ooze down the valley. If there is a situation where the simplex is trying to “pass through the eye of a needle”, it contracts itself in all directions, pulling itself in around its lowest (best) point.

In this implementation has been used an adaptive Nelder-Mead method algorithm (ANMA) which allows to change coefficients according to the number of dimensions





## Particle Swarm Optimization (PSO) method

<img align="right" heigth="340" width="340" src="https://upload.wikimedia.org/wikipedia/commons/e/ec/ParticleSwarmArrowsAnimation.gif">

It optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formula over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.

In this implementation a local topology has been used to avoid a too fast convergence.

Moreover a sort of correlation breaking has been implemented each 1000 iterations to improve the search and avoiding a stuck in a local minimum position.


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

### Execute
From the project directory
```console
python main.py
```

## Group Members

|  Reg No.  |  Name     |  Surname  |     Email                              |    Username      |
| :-------: | :-------: | :-------: | :------------------------------------: | :--------------: |
|   985203  | Salvatore | Pisciotta | `salvatore.pisciotta2@studio.unibo.it` | [_SalvoPisciotta_](https://github.com/SalvoPisciotta) |
|  1005271  | Giuseppe  | Boezio    | `giuseppe.boezio@studio.unibo.it`      | [_giuseppeboezio_](https://github.com/giuseppeboezio) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
