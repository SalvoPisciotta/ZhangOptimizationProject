import numpy as np

class NelderMeadSimplexOptimizer:
    reflection_coeff = 1.0
    expansion_coeff = 2.0
    contraction_coeff = 0.5
    shrinking_coeff = 0.5
    
    # <objective_function>: objective function, should match the specified dimension
    # <dimension>: dimension of parameter vector (integer)
    # <initial values>: list of <dimension+1> np.arrays of length <dimension> each
    # <stop_thresh>: float value stopping criterion, absolute of objective function value of best vs. worst vertex
    # <max_iter>: stopping criterion, maximum number of iterations
    def __init__(self, objective_function, dimension, initial_values, stop_thresh=1e-4, max_iter=500):
        self.obj_func = objective_function
        self.dimension = dimension
        self.vertices_and_values = []
        self.stop_thresh = stop_thresh
        self.max_iter = max_iter
        for vertex_iterator in initial_values:
            # create list of tuples (objective function value, vertex)
            self.vertices_and_values.append( (self.obj_func(vertex_iterator), vertex_iterator) )
    
    @staticmethod
    def create_random_vertices(dimension, center, scale):
        vertex_list = []
        rng = np.random.default_rng()
        for i in range(dimension+1):
            vertex_list.append( center + float(scale) * rng.random((dimension,)) )
        return vertex_list
    
    def calculate_centroid(self):
        # calculate center of all vertices except the worst
        self.centroid = np.zeros(self.dimension)
        for i in range(len(self.vertices_and_values)-1):
            self.centroid += self.vertices_and_values[i][1]
        self.centroid /= float( len(self.vertices_and_values) - 1 )
    
    def sort_vertices(self):
        # sort obj. func. values and their vertices by the obj. func. value
        self.vertices_and_values = sorted(self.vertices_and_values, key=lambda tup: tup[0])
        # store 2 best and 2 worst values and vertices separately
        self.best = self.vertices_and_values[0]
        self.second_best = self.vertices_and_values[1]
        self.second_worst = self.vertices_and_values[-2]
        self.worst = self.vertices_and_values[-1]
        
    def reflect(self):
        new_vertex = self.centroid * ( 1.0 + self.reflection_coeff ) - self.reflection_coeff * self.worst[1]
        new_obj_func_value = self.obj_func(new_vertex)
        return (new_obj_func_value, new_vertex)
        
    def expand(self):
        new_vertex = self.centroid * ( 1.0 + self.expansion_coeff ) - self.expansion_coeff * self.worst[1]
        new_obj_func_value = self.obj_func(new_vertex)
        return (new_obj_func_value, new_vertex)
         
    def contract(self, _vertex):
        new_vertex = self.centroid * ( 1.0 - self.contraction_coeff ) + self.contraction_coeff * _vertex
        new_obj_func_value = self.obj_func(new_vertex)
        return (new_obj_func_value, new_vertex)           
           
    def shrink(self):
        # iterate over all vertices except the best (first) one
        for i in range(1, len(self.vertices_and_values)):
            # shrink
            new_vertex = self.best[1] * (1.0 - self.shrinking_coeff) + self.shrinking_coeff * self.vertices_and_values[i][1]
            new_obj_func_value = self.obj_func(new_vertex)
            # replace vertices and new objective function values
            self.vertices_and_values[i] = (new_obj_func_value, new_vertex)
            
    def find_minimum(self, verbose=False):
        num_iterations = 0
        while(True):
            num_iterations += 1
            self.sort_vertices()
            self.calculate_centroid()
            # do reflection
            (reflection_value, reflection_vertex) = self.reflect()
            if( (reflection_value < self.second_worst[0]) and (reflection_value <= self.best[0]) ):
                # accept reflection, replace worst vertex by reflection
                self.vertices_and_values[-1] = (reflection_value, reflection_vertex)
                if(verbose):
                    print("reflection")
            elif( reflection_value < self.best[0] ):
                # do expansion
                (expansion_value, expansion_vertex) = self.expand()
                if( expansion_value <= reflection_value):
                    # accept expansion, replace worst vertex by expansion
                    self.vertices_and_values[-1] = (expansion_value, expansion_vertex)
                    if(verbose):
                        print("expansion")
                else:
                    # accept reflection, replace worst vertex by reflection
                    self.vertices_and_values[-1] = (reflection_value, reflection_vertex)
            elif( (reflection_value < self.worst[0]) and (reflection_value >= self.second_worst[0]) ):
                # do outside contraction towards reflection
                (outside_contraction_value, outside_contraction_vertex) = self.contract(reflection_vertex)
                if( outside_contraction_value <= reflection_value ):
                    # accept outside contraction, replace worst vertex by outside contraction
                    self.vertices_and_values[-1] = (outside_contraction_value, outside_contraction_vertex)
                    if(verbose):
                        print("outside contraction")
                else:
                    # shrink
                    self.shrink()
                    if(verbose):
                        print("shrink")
            else: # reflection_value >= self.worst[0]
                # do inside contraction towards worst
                (inside_contraction_value, inside_contraction_vertex) = self.contract(self.worst[1])
                if( inside_contraction_value <= self.worst[0]):
                    # accept inside contraction, replace worst vertex by inside contraction
                    self.vertices_and_values[-1] = (inside_contraction_value, inside_contraction_vertex)
                    if(verbose):
                        print("inside contraction")
                else:
                    # shrink
                    self.shrink()
                    if(verbose):
                        print("shrink")
            if(verbose):
                print("minimum:", self.best[0], "vertex:", self.best[1])    
            distance = abs(self.worst[0] - self.best[0])
            if(verbose):
                print("dist:", distance)
            if( distance < self.stop_thresh ):
                break
                
        self.sort_vertices()
        return (self.best)

