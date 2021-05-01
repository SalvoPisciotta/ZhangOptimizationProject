import numpy as np
from matplotlib import pyplot as plt
import time
import utils as ut

def print_loss(iteration_results,final_results,n_chessboard):
    '''
    iteration_results: list of the list containg the value of the loss function at each iteration for every chessboard
    final_results: list of the final loss function value for each chessboard
    n_chessboard: number of processed chessboard

    '''
    #plot of the descent of loss function for each chessboard
    for i in range(3):
        plt.plot(*zip(*iteration_results[i]),label = "Chessboard {}".format(i+5))
    plt.title('Comparsion between the descent of the loss of every chessboard analized')
    #plot settings
    plt.legend()
    plt.xlabel('Number of iterates')
    plt.ylabel('Loss function')
    plt.show()
    plt.ylabel('Final value of loss function')
    plt.title('Final value of the loss for every chessboard')
    #plot of the final values of the loss function for each chessboard
    plt.plot(*zip(*final_results),markersize=15,marker='o',linewidth=0)
    plt.tight_layout()
    plt.show()


def generate_starting_points(point, stepsize):
    '''
    point:  point from which it is generate the orthogonal base
    stepsize: lenght of the sides of simplex
    '''
    num = point.shape[0]
    identity = np.eye(num)
    starting_points = [point]
    #generation of initial simplex
    for i in range(num):
        starting_points.append(point + stepsize * identity[i,:].T)
    return starting_points


def centroid_calculation(simplex,loss_function,m,w):
    '''
    simplex: current simplex 
    loss_function: function of loss
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    '''
    centroid = np.zeros(len(simplex)-1)
    for i in range(len(simplex)-1):
        centroid += simplex[i][1]
    centroid /= float( len(simplex)-1 )
    centroid_value = loss_function(m,np.reshape(centroid,(3,3)),w)

    return (centroid_value,centroid)

def reflection(worst,centroid,coeff,loss_function,m,w):
    '''
    worst: tuple of the worst point of the simplex
    centroid: current centroid tuple of the simplex
    coeff: reflection coefficent
    loss_function: function of loss
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    '''
    reflection_point = centroid[1] * ( 1.0 + coeff ) - coeff * worst[1]
    reflection_value = loss_function(m, np.reshape(reflection_point,(3,3)) ,w)
    
    return (reflection_value, reflection_point)

def expansion(reflection,centroid,coeff,loss_function,m,w):
    '''
    reflection: reflected tuple of the simplex
    centroid: current centroid tuple of the simplex
    coeff: expansion coefficent
    loss_function: function of loss
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    '''
    expansion_point = centroid[1] * (1-coeff) + coeff*reflection[1]
    expansion_value = loss_function(m, np.reshape(expansion_point,(3,3)) ,w)

    return (expansion_value,expansion_point)

def outer_contraction(reflection,centroid,coeff,loss_function,m,w):
    '''
    reflection: reflected tuple of the simplex
    centroid: current centroid tuple of the simplex
    coeff: contraction coefficent
    loss_function: function of loss
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    '''
    contraction_point = centroid[1] + coeff * (reflection[1] - centroid[1])
    contraction_value = loss_function(m, np.reshape(contraction_point,(3,3)), w)
    return (contraction_value,contraction_point)

def inner_contraction(reflection,centroid,coeff,loss_function,m,w):
    '''
    reflection: reflected tuple of the simplex
    centroid: current centroid tuple of the simplex
    coeff: contraction coefficent
    loss_function: function of loss
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    '''
    contraction_point = centroid[1] - coeff * (reflection[1] - centroid[1])
    contraction_value = loss_function(m, np.reshape(contraction_point,(3,3)), w)
    return (contraction_value,contraction_point)

def shrink(simplex,coeff,loss_function,m,w):
    '''
    simplex: current simplex 
    loss_function: function of loss
    coeff: shrink coefficent
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    '''
    for i in range (1,len(simplex)):
        shrink_point = (simplex[0][1]+simplex[i][1])/2
        shrink_value = loss_function(m, np.reshape(shrink_point, (3,3)), w)
        simplex[i] = (shrink_value, shrink_point)
    return simplex


def nelder_mead_optimizer(loss_function, m, w, start ,max_it = 1e+10, max_fun_eval = 1e+10, toll_fun = 1e-6, toll_x = 1e-6):
    '''
    loss_function: function of loss
    m: image_points used as loss function parameter
    w: worlds point used as loss function parameter
    start: starting simplex 
    max_it: maximum number of iteration
    max_fun_eval: maximum number of evaluation of the loss function
    toll_fun: tollerance of the distance between best and worst of function value of the simplex
    toll_x: tollerance of the distance between worst and best point of the simplex
    '''
    #getting number of dimension
    num = start[0].shape[0]
    #adapting coefficients to the number of dimension
    #implementation of the ANMA variation of NMA
    reflect_coeff = 1
    exp_coeff = 1 + 2 / num
    contract_coeff = 0.75 - 1 / 2*num
    shrink_coeff = 1 - 1 / num
    #create list of tuples (loss function value, vertex)
    simplex_list = []
    for i in range(len(start)):
        simplex_list.append( (loss_function(m, np.reshape(start[i],(3,3)), w) , start[i] )  )
    #counter of iterations
    counter_it = 0
    #counter of function evaluations
    counter_fun_eval = 0
    #initialized to satisfy tollerance criterion
    best_tuple = (0, np.zeros(simplex_list[0][1].shape[0]))
    worst_tuple = (toll_fun + 1, np.zeros(simplex_list[0][1].shape[0]))
    #loss values at each iterate
    loss_values = []

    while ((worst_tuple[0]-best_tuple[0] > toll_fun or np.linalg.norm(worst_tuple[1]-best_tuple[1],np.inf) > toll_x)
            and counter_it <= max_it and counter_fun_eval <= max_fun_eval):
        
        #sorting wrt the loss_function value of vertices and assign the best/worst vertex to respectevely variables
        simplex_list = sorted(simplex_list, key= lambda pair: pair[0])
        best_tuple = simplex_list[0]
        second_worst_tuple = simplex_list[-2]
        worst_tuple = simplex_list[-1]

        #print first and multiple of 500 iterates
        if (counter_it == 0 or counter_it % 500 == 0):
            flag = True
        else:
            flag = False

        #find the centroid of the simplex
        centroid_tuple = centroid_calculation(simplex_list, loss_function, m, w)
        counter_fun_eval += 1

        #reflection
        reflection_tuple = reflection(worst_tuple,centroid_tuple,reflect_coeff,loss_function,m,w)
        counter_fun_eval += 1
        if(flag):
            print('SIMPLEX= {} '.format(simplex_list))
            print('CENTROID= {} '.format(centroid_tuple))
            print('REFLECTION = {}'.format(reflection_tuple))
            print("Best value = {} at iteration = {}".format(best_tuple[0],counter_it))
            print("Worst value = {} at iteration = {}".format(worst_tuple[0],counter_it))
            print("Second worst value = {} at iteration = {}".format(second_worst_tuple[0],counter_it))
            print("--------------------------------------------------")
        
        #save the best loss function value result
        loss_values.append(( counter_it, best_tuple[0]))


        #reflection evaluation 
        if(reflection_tuple[0] >= best_tuple[0] and reflection_tuple[0] < second_worst_tuple[0]):
            counter_fun_eval += 4
            #accept the reflection and impose the worst equal to the reflection_tuple
            simplex_list[-1] = reflection_tuple
        if(reflection_tuple[0] < best_tuple[0]):
            counter_fun_eval += 2
            #expansion
            expansion_tuple = expansion(reflection_tuple,centroid_tuple,exp_coeff,loss_function,m,w)
            counter_fun_eval += 1
            #expansion evaluation
            if(expansion_tuple[0] < reflection_tuple[0]):
                counter_fun_eval += 2
                #accept the expansion and impose the worst equal to the refletion_tuple
                simplex_list[-1] = expansion_tuple
            else:
                #accept the reflection and impose the worst equal to the reflection_tuple
                simplex_list[-1] = reflection_tuple

        #contraction conditions (inner and outer)
        if(reflection_tuple[0] >= second_worst_tuple[0]):
            counter_fun_eval += 2
            #outer contraction
            if (reflection_tuple[0] >= second_worst_tuple[0] and reflection_tuple[0] < worst_tuple[0]):
                counter_fun_eval += 4
                out_contraction_tuple = outer_contraction(reflection_tuple,centroid_tuple,contract_coeff,loss_function,m,w)
                counter_fun_eval += 1
                if out_contraction_tuple[0] <= reflection_tuple[0]:
                    counter_fun_eval += 2
                    simplex_list[-1] = out_contraction_tuple
                else:
                    simplex_list = shrink(simplex_list,shrink_coeff,loss_function,m,w)
                    counter_fun_eval += best_tuple[1].shape[0] - 1
            #inner contraction
            else:
                in_contraction_tuple = inner_contraction(reflection_tuple,centroid_tuple,contract_coeff,loss_function,m,w)
                if in_contraction_tuple[0] < worst_tuple[0]:
                    counter_fun_eval += 2
                    simplex_list[-1] = in_contraction_tuple
                else:
                    simplex_list = shrink(simplex_list,shrink_coeff,loss_function,m,w)
                    counter_fun_eval += best_tuple[1].shape[0] - 1
        counter_it += 1

    print("EXIT CONDITION:\n")
    print("Toll_fun: {}".format(worst_tuple[0]-best_tuple[0] > toll_fun))
    print("Toll x: {}".format(np.linalg.norm(worst_tuple[1]-best_tuple[1],np.inf) > toll_x))
    print("Num iterations: {}".format(counter_it <= max_it))
    print("Num fun evaluations: {}".format(counter_fun_eval <= max_fun_eval))
    print("Loss function value: {}".format(best_tuple[0]))
    print("Iterations: {}".format(counter_it-1))
    return simplex_list[0],loss_values

#Main
def main():
    #starting time
    start = time.time()
    #number of dimension
    DIM = 9
    #tau displacement
    TAU = 50
    #number of chessboard to analize
    n_chessboard = 3
    #set the name of the image file
    dirname='Chessboards/Chessboard'
    img_names=[dirname + str(i) + '.jpg' for i in range(5,8,1) ]
    print("analizing: {}".format(img_names))
    final_results = []
    iteration_results = []
    for img in img_names:
        #get m and w that represent respectively the image coordinates and the world coordinates already trasformed from R to P
        #it takes about 1 minute
        m , w = ut.process_corners(img)
        #zhang optimization step (minimization of the distance from real coordinates in image plan and the ones found by the corner detector)
        #generating starting points
        starting_points = generate_starting_points(np.ones(DIM), TAU)
        best_homography , loss_values = nelder_mead_optimizer(ut.loss_function,m,w,starting_points)
        #lista di tuple
        final_results.append((img ,loss_values[-1][1]))
        iteration_results.append(loss_values)
    computation_time = time.time() - start 
    print("Time: {}".format(computation_time))
    print_loss(iteration_results,final_results,n_chessboard)
    #function that prints the points of the image and the projection error refering to the optimal H adapting m and w matrices
    m = m[:,:2]
    w = np.reshape(best_homography[1],(3,3)) @ w.T
    w = w.T[:,:2]
    ut.print_correspondences(img,best_homography[1],best_homography[0],m,w)
     