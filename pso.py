import numpy as np
import time
import ZhangOptimization as zo
        
def particle_swarm_optimization(loss, m, w, bounds, c1, c2, num_par, vmax, tol = 1e-13, max_iter = 20000):

    np.random.seed(50)
    # getting initial particles and other related data
    dim, particle_pos, particle_val, particle_velocity, particle_best, swarm_best, local_best = random_inizialization(loss, m, w, bounds, num_par)
    # last best swarm position
    old_swarm = np.zeros(dim)
    # implementing constriction factor
    phi = c1 + c2
    chi = 2/(abs(2-phi-np.sqrt((phi**2)-(4*phi))))
    # number of iterations
    iter = 0
    # number of iteration for creating a conflict in the swarm
    count = 0
    while abs(loss(m, np.reshape(old_swarm, (3,3)), w) - loss(m, np.reshape(swarm_best, (3,3)), w)) > tol and iter < max_iter:

        iter += 1
        print("Step {}".format(iter))
        # time.sleep(0.100)

        count += 1 
        # conflict in the swarm
        if count >1000:
            print('Particles are too friendly! Creating conflict...')
            for j in range(num_par):
                particle_velocity[j,:] = [np.random.uniform(-abs(bounds[i][1]-bounds[i][0]),abs(bounds[i][1]-bounds[i][0])) for i in range(dim)]
            count=0 #reset iteration count

        for i in range(num_par):
            eps_1, eps_2 = np.random.uniform(0,1,2)
            particle_velocity[i,:] += (c1*eps_1*(particle_best[i,:]-particle_pos[i,:]))
            particle_velocity[i,:] += (c2*eps_2*(local_best[i,:]-particle_pos[i,:]))
            particle_velocity[i,:] = particle_velocity[i,:]*chi
            # velocity is too high
            if particle_velocity[i].any() > vmax :
                particle_velocity[i,:] = vmax 
            # updating the position
            particle_pos[i,:] += particle_velocity[i,:]

            particle_fitness = loss(m, np.reshape(particle_pos[i,:], (3,3)), w)
        
            # model implementation
            if particle_fitness < particle_val[i]:
                particle_best[i,:] = particle_pos[i,:]
                particle_val[i] = particle_fitness
                f_swarm_best = loss(m, np.reshape(swarm_best, (3,3)), w)
                if particle_fitness < f_swarm_best:
                    print("New swarm best found")
                    old_swarm_best = swarm_best[:]
                    swarm_best = particle_best[i,:].copy()
            
            local_best = get_local_best(particle_pos, particle_val, num_par)

            # my implementation
            '''particle_val[i] = loss(m, np.reshape(particle_pos[i,:], (3,3)), w)
            # the value of the new position is better than the current best value for this particle
            if particle_val[i] < loss(m, np.reshape(particle_best[i], (3,3)), w):
                particle_best[i,:] = particle_pos[i,:]
                if particle_val[i] < loss(m, np.reshape(swarm_best, (3,3)), w):
                    old_swarm = swarm_best
                    swarm_best = particle_pos[i,:]'''

    return (loss(m, np.reshape(swarm_best, (3,3)), w), swarm_best)

# random initializion of the swarm
def random_inizialization(f, m, w, bounds, num_par):
    
    # getting number of dimension
    dim = len(bounds)
    # position of the particles
    particle_pos = np.zeros(num_par)
    particle_pos = particle_pos.tolist()
    # velocity of the particles
    particle_velocity = particle_pos[:]
    # current value of the particle
    particle_val = particle_pos[:]
    np.random.seed(50)
    for j in range(num_par):
        # starting points for particles
        particle_pos[j] = [np.random.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)]
        # computing the fitness function
        particle_val[j] = f(m, np.reshape(particle_pos[j], (3,3)), w)
        # velocity for each particle
        particle_velocity[j]=[np.random.uniform(-abs(bounds[i][1]-bounds[i][0]) ,abs(bounds[i][1]-bounds[i][0])) for i in range(dim)]
    
    # getting local best position using lbest ring topology
    local_best = get_local_best(particle_pos, particle_val, num_par)
    # getting swarm best position
    swarm_best = particle_pos[np.argmin(particle_val)]
    #setting all particles current positions to best
    particle_best = particle_pos.copy()

    return dim, np.array(particle_pos), np.array(particle_val), np.array(particle_velocity), np.array(particle_best), np.array(swarm_best), np.array(local_best)

# implementing local best topology
def get_local_best(particle_pos, particle_val, num_par):
    # creation of an empty list
    local_best = [0] * num_par
    # for each particle finding the best position looking at its neighbours (ring topology)
    for j in range(num_par):
        local_values = np.zeros(3)
        local_values[0] = particle_val[j-2]
        local_values[1] = particle_val[j-1]
        local_values[2] = particle_val[j]
        min_index = int(np.argmin(local_values))
        local_best[j-1] = particle_pos[min_index+j-2][:]
    return np.array(local_best)

if __name__ == '__main__':
    start = time.time()
    # Number of dimension
    DIM = 9
    # Set the name of the image file
    img = 'Chessboard.jpg'
    # Get m and w that represent respectively the image coordinates and the world coordinates already trasformed from R to P
    # It takes about 1 minute
    m , w = zo.process_corners(img)
    # Zhang optimization step (minimization of the distance from real coordinates in image plan and the ones found by the corner detector)
    # generating bounds for initial point
    dimension_bounds = [0,1000]
    bounds = []
    for i in range(DIM):
        bounds.append(dimension_bounds)
    # computing vmax
    vmax = (dimension_bounds[1]-dimension_bounds[0])*0.75
    # parameters have been get consulting a scientific paper
    c1 = -0.2746
    c2 = 4.8976
    num_par = 53
    best_homography = particle_swarm_optimization(zo.loss_function, m, w, bounds, c1, c2, num_par, vmax)
    #Function that prints the points of the image and the projection error refering to the optimal H
    m = m[:,:2]
    w = np.reshape(best_homography[1],(3,3)) @ w.T
    w = w.T[:,:2]
    zo.print_correspondences(img,best_homography[1],best_homography[0],m,w)
    print("Time: {}".format(time.time() - start)) 