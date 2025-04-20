import numpy as np
from scipy.optimize import linprog, minimize
import AllegroHandEnv
import dm_control
import mujoco as mj
import grasp_synthesis
import types

"""
Note: this code gives a suggested structure for implementing grasp synthesis.
You may decide to follow it or not. 
"""

def synthesize_grasp(env: grasp_synthesis.AllegroHandEnv, 
                         q_h_init: np.array,
                         fingertip_names: list[str], 
                         max_iters=1000, 
                         lr=0.1):
    """
    Given an initial hand joint configuration, q_h_init, return adjusted joint angles that are touching
    the object and approximate force closure. This is algorithm 1 in the project specification.

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h_init: array of joint positions for the hand
    max_iters: maximum number of iterations for the optimization
    lr: learning rate for the gradient step

    Output
    ------
    New joint angles after contact and force closure adjustment
    """
    #YOUR CODE HERE

    # Initialize the hand configuration and other parameters
    q_h = q_h_init
    iter_count = 0
    in_contact = False
    while iter_count < max_iters: # Start optimization loop
        iter_count += 1
        in_contact = env.all_fingers_touching(env, fingertip_names) # Check if all fingers are touching the object
        f = joint_space_objective(env, q_h, fingertip_names, in_contact)  # Compute objective function based on current q_h
        q_h_new = q_h - lr * numeric_gradient(f, q_h, env, fingertip_names, in_contact)  # Compute the gradient and update q_h
        
        in_contact = env.all_fingers_touching(env, fingertip_names)

        improvement = f - joint_space_objective(env, q_h_new, fingertip_names, in_contact)
        if improvement > 0:
            q_h = q_h_new
            if improvement < 1e-6:
                break
    return q_h

def joint_space_objective(env: grasp_synthesis.AllegroHandEnv, 
                          q_h: np.array,
                          fingertip_names: list[str], 
                          in_contact: bool, 
                          Q_plus_thresh,
                          beta=10.0, 
                          friction_coeff=0.5, 
                          num_friction_cone_approx=4):
    """
    This function minimizes an objective such that the distance from the origin
    in wrench space as well as distance from fingers to object surface is minimized.
    This is algorithm 2 in the project specification. 

    Parameters
    ----------
    env: AllegroHandEnv instance (can use to access physics)
    q_h: array of joint positions for the hand
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    beta: weight coefficient on the surface penalty 
    friction_coeff: Friction coefficient for the ball
    num_friction_cone_approx: number of approximation vectors in the friction cone

    Output
    ------
    fc_loss + (beta * d) as written in algorithm 2
    """
    env.set_configuration(q_h)
    #YOUR CODE HERE
    positions = env.get_contact_positions(fingertip_names)
    ball_radius = 0.05
    ball_center = env.physics.data.xpos["ball"]
    D = env.sphere_surface_distance(positions, ball_center, ball_radius)

    if not in_contact: return beta * D
    else: # in contact
        normals = env.get_contact_normals(env.physics.data.contact)
        FC = build_friction_cones(normals) # Step 4: Build friction cones
        G = build_grasp_matrix(positions, FC) # Step 5: Build grasp map (G)
        fc_loss = optimize_necessary_condition(G, env) # Step 6: Minimize the necessary condition for force closure
        if fc_loss < Q_plus_thresh:
            fc_loss = optimize_sufficient_condition(G)
        return fc_loss + beta * D

def numeric_gradient(function: types.FunctionType, 
                     q_h: np.array, 
                     env: grasp_synthesis.AllegroHandEnv, 
                     fingertip_names: list[str], 
                     in_contact: bool, 
                     eps=0.01):
    """
    This function approximates the gradient of the joint_space_objective

    Parameters
    ----------
    function: function we are taking the gradient of
    q_h: joint configuration of the hand 
    env: AllegroHandEnv instance 
    fingertip_names: names of the fingertips as defined in the MJCF
    in_contact: helper variable to determine if the fingers are in contact with the object
    eps: hyperparameter for the delta of the gradient 

    Output
    ------
    Approximate gradient of the inputted function
    """
    baseline = function(q_h, env, fingertip_names, in_contact)
    grad = np.zeros_like(q_h)
    for i in range(len(q_h)):
        q_h_pert = q_h.copy()
        q_h_pert[i] += eps
        val_pert = function(q_h_pert, env, fingertip_names, in_contact)
        grad[i] = (val_pert - baseline) / eps
    return grad


def build_friction_cones(normals: np.array, mu=0.5, num_approx=4):
    """
    This function builds a discrete friction cone around each normal vector. 

    Parameters
    ----------
    normal: nx3 np.array where n is the number of normal directions
        normal directions for each contact
    mu: friction coefficient
    num_approx: number of approximation vectors in the friction cone

    Output
    ------
    friction_cone_vectors: array of discretized friction cones represented 
    as vectors
    """
    #YOUR CODE HERE
    friction_cones = []
    for n in normals:
        friction_cone_vectors = []
        angle = np.arctan(mu)
        for i in range(num_approx):
            theta = i * 2 * np.pi / num_approx  # Evenly distributed in azimuthal angle
            # Use spherical coordinates to sample the friction cone
            x = np.cos(theta) * np.sin(angle)
            y = np.sin(theta) * np.sin(angle)
            z = np.cos(angle)

            rotation_matrix 
            vector = rotation_matrix @ np.array([x, y, z])
            friction_cone_vectors.append(vector)
        friction_cones.append(friction_cone_vectors)
    
    return np.array(friction_cone_vectors)


def build_grasp_matrix(positions: np.array, friction_cones: list, origin=np.zeros(3)):
    """
    Builds a grasp map containing wrenches along the discretized friction cones. 

    Parameters
    ----------
    positions: nx3 np.array of contact positions where n is the number of contacts
    firction_cone: a list of lists as outputted by build_friction_cones. 
    origin: the torque reference. In this case, it's the object center.
    
    Return a 2D numpy array G with shape (6, sum_of_all_cone_directions).
    """
    #YOUR CODE HERE
    alpha = 1 # might need to change later
    G = []

    for i, fc in enumerate (friction_cones): # extract first cone
        di = np.zeros ((3, 1))

        for dij in fc: 
            di += alpha * dij

        contact_pos = (positions[i] - origin) 
        torque = np.cross(contact_pos, di)
        wrench = np.concatenate((di, torque), axis=0)
        G.append(wrench)
    return np.array(G)



def optimize_necessary_condition(G: np.array, env: grasp_synthesis.AllegroHandEnv):
    """
    Returns the result of the L2 optimization on the distance from wrench origin to the
    wrench space of G

    Parameters
    ----------
    G: grasp matrix
    env: AllegroHandEnv instance (can use to access physics)

    Returns the minimum of the objective

    Hint: use scipy.optimize.minimize
    """
    #YOUR CODE HERE
    def objective():
        pass

    x0 = ...
    bounds = ...

    res = minimize(objective, x0, method='SLSQP', bounds=bounds)

    return res.fun


def optimize_sufficient_condition(G: np.array, K=20):
    """
    Runs the optimization from the project spec to evaluate Q- distance. 

    Parameters
    ----------
    G: grasp matrix
    K: number of approximations to the norm ball

    Returns the Q- value

    Hints:
        -Use scipy.optimize.linprog
        -Here's a resource with the basics: https://realpython.com/linear-programming-python/
        -You'll have to find a way to represent the alpha's for the constraints
            -Consider including the alphas in the linprog objective with coefficients 0 
        -For the optimization method, do method='highs'
    """
    #YOUR CODE HERE