import numpy as np
import dm_control
import mujoco as mj

class AllegroHandEnv:
    def __init__(self, physics: dm_control.mjcf.physics.Physics, 
                 q_h_slice: slice, 
                 object_name: str, 
                 num_fingers=4):
        
        self.physics = physics
        self.q_h_slice = q_h_slice
        self.num_fingers = num_fingers
        self.object_name = object_name


    def set_configuration(self, q_h: np.array):
        self.physics.data.qpos[self.q_h_slice] = q_h
        self.physics.forward()

    def all_fingers_touching(self, fingertip_names: list[str]):
        """
        Returns True if all fingers are touching the object, False otherwise
        """
        contact_list = self.physics.data.contact
        fingers_touching = [False] * self.num_fingers
        for contact in contact_list:
            if self.object_name in contact.geom1.name or self.object_name in contact.geom2.name: # Check if object in contact
                for i, fingertip_name in enumerate(fingertip_names): # Check if fingertip in contact
                    if fingertip_name in contact.geom1.name or fingertip_name in contact.geom2.name:
                        fingers_touching[i] = True

        # Return True if all fingers are touching the object
        return all(fingers_touching)

    def get_contact_positions(self, body_names: list[str]):
        """
        Input: list of the names in the XML of the bodies that are in contact
        Returns: (num_contacts x 3) np.array containing 
        finger positions in workspace coordinates
        """
        #YOUR CODE HERE
        contact_positions = []
        contact_list = self.physics.data.contact

        # Iterate through each contact
        for contact in contact_list:
            # Check if the contact involves any of the specified body names
            for body_name in body_names:
                if body_name in contact.geom1.name or body_name in contact.geom2.name:
                    # Get the contact point's position in the world coordinate system
                    contact_pos = contact.pos  # The contact position is stored in contact.pos

                    # Append the contact position to the list
                    contact_positions.append(contact_pos)

        # Convert the list of contact positions to a numpy array (num_contacts x 3)
        return np.array(contact_positions)


    def get_contact_normals(self, contact: mj._structs._MjContactList):
        """
        Input: contact data structure that contains MuJoCo contact information
        Returns the normal vector for each geom that's in contact with the ball
        
        Tip: 
            -See information about the mjContact_ struct here: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact
            -Get normals for all the geoms in contact with the ball, not just the fingertips
        """
        contact_normals = []
        for c in contact:
            contact_normals.append(c.normal)
        return np.array(contact_normals)

class AllegroHandEnvSphere(AllegroHandEnv):
    def __init__(self, physics: dm_control.mjcf.physics.Physics, 
                 sphere_center: int, 
                 sphere_radius: int, 
                 q_h_slice: slice, 
                 object_name: str):
        super().__init__(physics, q_h_slice, object_name)
        self.physics = physics
        self.sphere_center = sphere_center
        self.sphere_radius = sphere_radius
        self.q_h_slice = q_h_slice
        self.num_fingers = 4
    
    def sphere_surface_distance(self, pos: np.array, center: np.array, radius: int):
        """
        Returns the distance from pos to the surface of a sphere with a specified
        radius and center
        """
        d = np.linalg.norm(pos - center) - radius
        return d