import numpy as np
import dm_control
from dm_control import mjcf
from scipy.spatial.transform import Rotation as R 

def clip_to_valid_state(physics: dm_control.mjcf.physics.Physics, qpos: np.array):
    """
    This function returns qpos with every value clipped to the allowable joint range as specified
    in the MJCF. 
    """
    qpos_clipped = qpos.copy()

    for joint_idx in range(physics.model.njnt):
        joint_range = physics.model.jnt_range[joint_idx]

        qpos_clipped[physics.model.jnt_qposadr[joint_idx]] = np.clip(
            qpos_clipped[physics.model.jnt_qposadr[joint_idx]], 
            joint_range[0],
            joint_range[1])

    return qpos_clipped

def quaternion_error_naive(current_quat: np.array, target_quat: np.array):
    """
    Rough orientation error between two quaternions.
    This is just a rough measure and doesn't work well for
    large angles, so you might want to consider using
    something more advanced to compare quaternions.
    """
    q_diff = quat_multiply(target_quat, quat_conjugate(current_quat))
    return q_diff[1:4]  # ignoring w

def quat_multiply(q1: np.array, q2: np.array):
    """Multiply two quaternions, returning q1*q2 in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])

def quat_conjugate(q: np.array):
    """Return quaternion conjugate: [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def vector_to_quaternion(normal_vector):

    reference_vector = np.array([0, 0, 1]) # Step 1: Define the reference vector (usually the z-axis)
    axis_of_rotation = np.cross(reference_vector, normal_vector) # Step 2: Compute the axis of rotation (cross product)

    # Step 3: Compute the angle of rotation (dot product and arccos)
    cos_theta = np.dot(reference_vector, normal_vector)
    angle = np.arccos(cos_theta)

    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation) # Step 4: Normalize the axis of rotation (to ensure it's a unit vector)

    # Step 5: Create the quaternion using scipy.spatial.transform
    rotation = R.from_rotvec(axis_of_rotation * angle)  # axis * angle
    quaternion = rotation.as_quat()  # Quaternion [x, y, z, w]
    
    return quaternion
