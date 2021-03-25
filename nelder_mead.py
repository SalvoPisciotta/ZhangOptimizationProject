import numpy as np
from numpy import linalg as la
import ZhangOptimization as zo
from scipy.optimize import rosen

#NOTE:
#Access to value = print(simplex_list[0][0])
#Access to vector = print(simplex_list[0][1])
#Understand how to generate the starting values

#start = [np.random.rand(9) for i in range(10)]
#w = np.random.rand(5,3)
#m = np.random.rand(5,3)

#Function for print only the values of the list of tuples
def printing(l):
    print('Value')
    for i in range(len(l)):
        print('{}'.format(l[i][0] , l[i][1]))

def centroid_calculation(simplex,loss_function,m,w):
    centroid = np.zeros(len(simplex)-1)
    for i in range(len(simplex)-1):
        centroid += simplex[i][1]
    centroid /= float( len(simplex)-1 )
    #centroid_value = loss_function(m,np.reshape(centroid,(3,3)),w)
    centroid_value = loss_function(centroid)

    return (centroid_value,centroid)

def reflection(worst,centroid,coeff,loss_function,m,w):
    reflection_point = centroid[1] * ( 1.0 + coeff ) - coeff * worst[1]
    #reflection_value = loss_function(m, np.reshape(reflection_point,(3,3)) ,w)
    reflection_value = loss_function(reflection_point)
    return (reflection_value, reflection_point)

def expansion(reflection,centroid,coeff,loss_function,m,w):
    expansion_point = centroid[1] * (1-coeff) + coeff*reflection[1]
    #expansion_value = loss_function(m, np.reshape(expansion_point,(3,3)) ,w)
    expansion_value = loss_function(expansion_point)
    return (expansion_value,expansion_point)

def contraction(worst,centroid,coeff,loss_function,m,w):
    contraction_point = centroid[1] * (1-coeff) + coeff*worst[1]
    #contraction_value = loss_function(m, np.reshape(contraction_point,(3,3)), w)
    contraction_value = loss_function(contraction_point)
    return (contraction_value,contraction_point)

def shrink(simplex,coeff,loss_function,m,w):
    for i in range (1,len(simplex)):
        shrink_point = (simplex[0][1]+simplex[i][1])/2
        #shrink_value = loss_function(m, np.reshape(shrink_point, (3,3)), w)
        shrink_value = loss_function(shrink_point)
        simplex[i] = (shrink_value, shrink_point)
    return simplex


def nelder_mead_optimizer(loss_function,m,w,start,max_it = 50,toll = 10e-6,reflect_coeff = 1.0,exp_coeff = 2.0,contract_coeff = 0.5,shrink_coeff = 0.5):
    #Create list of tuples (loss function value, vertex)
    simplex_list = []
    for i in range(len(start)):
        #simplex_list.append( (loss_function(m, np.reshape(start[i],(3,3)), w) , start[i] )  )
        simplex_list.append( (loss_function(start[i]), start[i]))

    counter_it = 0
    best_value = 1

    while(counter_it<=max_it and best_value >= toll):
        counter_it += 1

        #Sorting wrt the loss_function value of vertices and assign the best/worst vertex to respectevely variables
        simplex_list = sorted(simplex_list, key= lambda pair: pair[0])
        best_tuple = simplex_list[0]
        second_worst_tuple = simplex_list[-2]
        worst_tuple = simplex_list[-1]

        print("Best value = {} at iteration = {}".format(best_tuple[0],counter_it))
        print("Worst value = {} at iteration = {}".format(worst_tuple[0],counter_it))
        print("Second worst value = {} at iteration = {}".format(second_worst_tuple[0],counter_it))

        #Find the centroid of the simplex
        centroid_tuple = centroid_calculation(simplex_list,loss_function,m,w)

        #Reflection
        reflection_tuple = reflection(worst_tuple,centroid_tuple,reflect_coeff,loss_function,m,w)
        #Reflection evaluation
        if( reflection_tuple[0] >= best_tuple[0] and reflection_tuple[0] < second_worst_tuple[0] ):
            #accept the reflection and impose the worst equal to the reflection_tuple
            simplex_list[-1] = reflection_tuple
            print("reflection")
        elif( reflection_tuple[0] < best_tuple[0]):
            #Expansion
            expansion_tuple = expansion(reflection_tuple,centroid_tuple,exp_coeff,loss_function,m,w)
            #Expansion evaluation
            if(expansion_tuple[0] < reflection_tuple[0]):
                #accept the expansion and impose the worst equal to the refletion_tuple
                simplex_list[-1] = expansion_tuple
                print("expansion")
            else:
                #accept the reflection and impose the worst equal to the reflection_tuple
                simplex_list[-1] = reflection_tuple
                print("reflection")
        elif(reflection_tuple[0]<worst_tuple[0] and reflection_tuple[0] >= second_worst_tuple[0]):
            #Contraction
            contraction_tuple = contraction(worst_tuple,centroid_tuple,contract_coeff,loss_function,m,w)
            #Contraction evaluation
            if(contraction_tuple[0] < worst_tuple[0]):
                #accept the contraction and impose the worst equal to the contraction_tuple
                simplex_list[-1] = contraction_tuple
                print("contraction")
            else:
                #Shrink and update the simplex_list
                simplex_list = shrink(simplex_list,shrink_coeff,loss_function,m,w)

    return simplex_list[0]


# Testing the function
start = ([ np.array([1.0,3.0]), np.array([2.0,7.0]), np.array([5.0,2.0]) ])
solution = nelder_mead_optimizer(rosen,[],[],start)
print(solution)