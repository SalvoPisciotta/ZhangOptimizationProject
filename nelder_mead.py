import numpy as np
from numpy import linalg as la
import ZhangOptimization as zo

#NOTE:
#Access to value = print(simplex_list[0][0])
#Access to vector = print(simplex_list[0][1])
#Understand how to generate the starting values

start = [np.random.rand(9) for i in range(10)]
w = np.random.rand(5,3)
m = np.random.rand(5,3)

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
    centroid_value = loss_function(m,np.reshape(centroid,(3,3)),w)

    return (centroid_value,centroid)

def reflection(worst,centroid,coeff,loss_function,m,w):
    reflection_point = centroid[1] * ( 1.0 + coeff ) - coeff * worst[1]
    reflection_value = loss_function(m, np.reshape(reflection_point,(3,3)) ,w)
    return (reflection_value, reflection_point)


def nelder_mead_optimizer(loss_function,m,w,start,max_it = 50,toll = 10e-6,reflect_coeff = 1.0,exp_coeff = 2.0,contract_coeff = 0.5,shrink_coeff = 0.5):
    #Create list of tuples (loss function value, vertex)
    simplex_list = []
    for i in range(len(start)):
        simplex_list.append( (loss_function(m, np.reshape(start[i],(3,3)), w) , start[i] )  )

    counter_it = 0
    best_value = 1

    while(counter_it<=max_it and best_value >= toll):
        counter_it += 1

        #Sorting wrt the loss_function value of vertices and assign the best/worst vertex to respectevely variables
        simplex_list = sorted(simplex_list, key= lambda pair: pair[0])
        best_tuple = simplex_list[0]
        worst_tuple = simplex_list[len(simplex_list)-1]

        #Find the centroid of the simplex
        centroid_tuple = centroid_calculation(simplex_list,loss_function,m,w)

        #Reflection
        reflection_tuple = reflection(worst_tuple,centroid_tuple,reflect_coeff,loss_function,m,w)

        




    return True

nelder_mead_optimizer(zo.loss_function,m,w,start)
