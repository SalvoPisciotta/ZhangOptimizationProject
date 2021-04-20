import numpy as np
import time
import ZhangOptimization as zo

# get the initial best position for each particle
'''
def get_particle_best(particle_pos,particle_val, num_par):
    # number of near position to consider
    NUM_NEIGHBOURS = 3
    particle_best = np.empty(num_par)
    # finding the best position for each particle
    for j in range(num_par):
        local_vals = np.zeros(NUM_NEIGHBOURS)
        local_vals[0] = particle_pos_val[j-2]
        local_vals[1] = particle_pos_val[j-1]
        local_vals[2] = particle_pos_val[j]
        min_index=int(np.argmin(local_vals))
        local_best[j-1]=particle_pos[min_index+j-2][:]
    return np.array(local_best)

    return 0
'''


def random_inizialization(f, m, w, bounds, num_par):

    # getting number of dimension
    dim = len(bounds)
    particle_pos = np.zeros(num_par)
    particle_pos = particle_pos.tolist()
    particle_velocity = particle_pos[:]
    particle_val = particle_pos[:]

    for j in range(num_par):
        # starting points for particles
        particle_pos[j] = [np.random.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)]
        # computing the fitness function
        particle_val[j] = f(m, np.reshape(particle_pos[j], (3,3)), w)
        # velocity for each particle
        particle_velocity[j]=[np.random.uniform(-abs(bounds[i][1]-bounds[i][0]) ,abs(bounds[i][1]-bounds[i][0])) for i in range(dim)]
    # best position for each particle
    particle_best = particle_pos.copy()
    # best global position
    swarm_best = particle_pos[np.argmin(particle_val)]

    return dim, np.array(particle_pos), np.array(particle_val), np.array(particle_velocity), np.array(particle_best), np.array(swarm_best)
        
def particle_swarm_optimization(loss, m, w, bounds, omega, phi_p, phi_g, num_par, tol = 1e-10):

    # getting initial particles and other related data
    dim, particle_pos, particle_val, particle_velocity, particle_best, swarm_best = random_inizialization(loss, m, w, bounds, num_par)

    # last best global point
    old_swarm = np.zeros(dim)

    while abs(loss(m, np.reshape(old_swarm, (3,3)), w) - loss(m, np.reshape(swarm_best, (3,3)), w)) > tol:
        for i in range(num_par):
            r_p, r_g = np.random.uniform(0,1,2)
            particle_velocity[i,:] += (phi_p * r_p * (particle_best[i,:] - particle_pos[i,:]))
            particle_velocity[i,:] += (phi_g * r_g * (swarm_best[i,:] - particle_pos[i,:]))
            particle_velocity[i,:] = particle_velocity[i,:] * omega
            #if particle_velocity[i].any() > vmax : #is any velocity is greater than vmax
                    #particle_velocity[i,:]=vmax #set velocity to vmax
            #all of the above is regarding updating the particle's velocity
            #with regards to various parameters (local_best, p_best etc..)
            # updating position
            particle_pos[i,:] += particle_velocity[i,:]

            particle_val[i] = loss(m, np.reshape(particle_pos[i,:], (3,3)), w)
            # the value of the new position is better than the current best value for this particle
            if particle_val[i] < loss(m, np.reshape(particle_best[i], (3,3)), w):
                particle_best[i,:] = particle_pos[i,:]
                if particle_val[i] < loss(m, np.reshape(swarm_best, (3,3)), w):
                    old_swarm = swarm_best
                    swarm_best = particle_pos[i,:]

    return (loss(m, np.reshape(swarm_best, (3,3)), w), swarm_best)

if __name__ == '__main__':
    start = time.time()
    # Number of dimension
    DIM = 9
    # Set the name of the image file
    img = 'Chessboard.jpg'
    # Get m and w that represent respectively the image coordinates and the world coordinates already trasformed from R to P
    # It takes about 1 minute
    m , w = zo.process_corners(img)
    m = m[:8,:]
    w = w[:8,:]
    # Zhang optimization step (minimization of the distance from real coordinates in image plan and the ones found by the corner detector)
    # generating bounds for initial point
    bounds = []
    for i in range(DIM):
        bounds.append([0,100])
    # parameters have been get consulting a scientific paper
    omega = -0.3488
    phi_p = -0.2746
    phi_g = 4.8976
    num_par = 53
    best_homography = particle_swarm_optimization(zo.loss_function, m, w, bounds, omega, phi_p, phi_g, num_par)
    #Function that prints the points of the image and the projection error refering to the optimal H
    m = m[:,:2]
    w = np.reshape(best_homography[1],(3,3)) @ w.T
    w = w.T[:,:2]
    zo.print_correspondences(img,best_homography[1],best_homography[0],m,w)
    print("Time: {}".format(time.time() - start)) 