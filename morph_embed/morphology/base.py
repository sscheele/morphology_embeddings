import numpy as np
from lxml import etree
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Literal
from morph_embed.setup_logger import logger
from morph_embed.morphology.mjcf_builder import MJCFBuilder

import mujoco
import mujoco.viewer

LINK_RADIUS = 0.03

@dataclass
class Joint:
    name: str

@dataclass
class HingeJoint(Joint):
    axis: np.ndarray
    range: Tuple[float, float] = (-1.57, 1.57)

@dataclass
class FixedJoint(Joint):
    translation: np.ndarray
    rotation: np.ndarray

@dataclass
class Link:
    name: str
    length: float
    radius: float
    children: List['Link'] = field(default_factory=lambda: [])
    joint: Optional[Joint] = None

def random_unit_vector(size=3):
    v = np.random.uniform(size=size)
    return v / np.linalg.norm(v)

def random_rotation_quat():
    axis = random_unit_vector()
    angle = np.random.uniform(0, 2 * np.pi)
    s = np.sin(angle / 2)
    return [axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle/2)]

class Morphology:
    # Class attributes for morphology parameters
    num_links_range = (3, 7)  # Range for number of links in the chain
    length_range = (0.1, 0.3)

    def __init__(self, params=None):
        if params is None:
            # Default parameters for random generation
            self.params = {}
            self._generate_random_morphology()
        else:
            self.params = params
            if 'links' in params:
                self._load_morphology(params['links'])
            else:
                self._generate_random_morphology()
        
        # Initialize end_effectors
        self._compute_end_effectors()

    def _generate_random_morphology(self):
        """Generate a random morphology using the current parameters."""
        self.base = Link(
            name="base",
            length=0,
            radius=0
        )
        
        current_link = self.base
        num_links = np.random.randint(*self.num_links_range)
        for i in range(num_links):
            length = np.random.uniform(*self.length_range)
            radius = LINK_RADIUS
            
            # Create joint and link
            joint = HingeJoint(
                name=f"j_link_{i}",
                axis=random_unit_vector()
            )
            
            new_link = Link(
                name=f"link_{i}",
                length=length,
                radius=radius,
                joint=joint
            )
            
            current_link.children = [new_link]
            current_link = new_link

    def _compute_end_effectors(self):
        """
        Compute the end effectors for this morphology.
        
        For random morphologies, links with no children are considered end effectors.
        For non-random morphologies loaded from params, end effectors are loaded from params
        if available, otherwise determined by links with no children.
        """
        # Check if end effectors are specified in params
        if 'end_effectors' in self.params:
            self.end_effectors = self.params['end_effectors']
            return
            
        # Otherwise, find links with no children
        self.end_effectors = []
        
        def find_end_effectors(link: Link):
            if len(link.children) == 0 and link.name != "base":
                self.end_effectors.append(link.name)
            else:
                for child in link.children:
                    find_end_effectors(child)
        
        # Start traversal from the base link
        find_end_effectors(self.base)
    
    def _load_morphology(self, links_data: Dict):
        """Load a saved morphology from the provided data."""
        def create_link_from_dict(data: Dict) -> Link:
            link = Link(
                name=data['name'],
                length=data['length'],
                radius=data['radius']
            )
            
            if 'joint' in data:
                joint_data = data['joint']
                if 'type' not in joint_data or joint_data['type'] == 'hinge':
                    link.joint = HingeJoint(
                        name=joint_data['name'],
                        axis=np.array(joint_data['axis']),
                        range=tuple(joint_data.get('range', (-1.57, 1.57)))
                    )
                elif joint_data['type'] == 'fixed':
                    link.joint = FixedJoint(
                        name=joint_data['name'],
                        translation=joint_data['translation'],
                        rotation=joint_data['rotation']
                    )
            
            if 'children' in data:
                link.children = create_link_from_dict(data['children'])
            
            return link
        
        self.base = create_link_from_dict(links_data)
    
    @staticmethod
    def _get_links_chain(start_link: Link) -> List[Link]:
        """Flattens the link tree via depth-first traversal"""
        links = [start_link]
        if len(start_link.children) == 0:
            return links

        for c in start_link.children:
            links += Morphology._get_links_chain(c)
        return links

    def to_mjcf(self, enable_gravity: bool = False) -> str:
        """
        Convert the morphology to MJCF format string.
        
        Args:
            enable_gravity: Whether to enable gravity in the simulation (default: False)
        
        Returns:
            str: The MJCF XML string representing the morphology
        """        
        # Get the links chain
        links_chain = self._get_links_chain(self.base)
        
        # Use the builder to create the MJCF
        return MJCFBuilder.build_mjcf(
            links_chain=links_chain,
            enable_gravity=enable_gravity,
            model_name="morphology",
            prefix=""
        )

    def get_params(self) -> Dict:
        """Get the current parameters of the morphology."""
        return self.params.copy()

    def to_dict(self) -> Dict:
        """Convert the morphology to a dictionary representation."""
        def link_to_dict(link: Link) -> Dict:
            data = {
                'name': link.name,
                'length': link.length,
                'radius': link.radius,
            }
            
            if link.joint:
                data['joint'] = {
                    'name': link.joint.name,
                    'axis': link.joint.axis.tolist(),
                    'range': link.joint.range,
                    'type': link.joint.type
                }
            
            if len(link.children) > 0:
                data['children'] = [link_to_dict(c) for c in link.children]
            else:
                data['children'] = []
            
            return data
        
        # Include end effectors in the dictionary representation
        result = {
            'links': link_to_dict(self.base),
            'end_effectors': self.end_effectors
        }
            
        return result

def quat2mat(quat):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def mujoco_fk(data, model, link):
    """
    Calculate the tip position of a link using forward kinematics.
    
    Args:
        data: MuJoCo data object
        model: MuJoCo model object
        link: Link object
        
    Returns:
        np.ndarray: 3D position of the tip of the link
    """
    # Get the link's body ID and root position
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link.name)
    link_root_pos = data.xpos[link_id].copy()
    
    # Get the link's orientation as a quaternion
    link_quat = data.xquat[link_id].copy()
    
    # Calculate the tip position by adding the link's length along the z-axis
    rot_mat = quat2mat(link_quat)
    z_axis = rot_mat[:, 2]  # Third column is the z-axis
    tip_pos = link_root_pos + z_axis * link.length
    
    return tip_pos

def print_link_structure(link, depth=0):
    indent = "  " * depth
    if link.name != "base":
        logger.info(f"{indent}- {link.name}: length={link.length}, radius={link.radius}")
    if len(link.children) > 0:
        for c in link.children:
            print_link_structure(c, depth + 1)

def test_fk(num_timesteps: int = 100, control_magnitude: float = 0.1) -> np.ndarray:
    """
    Generate a random morphology, add small control values, simulate its dynamics,
    and return the end effector position.
    
    Args:
        num_timesteps: Number of simulation timesteps to run
        control_magnitude: Magnitude of the random control values
        
    Returns:
        np.ndarray: The 3D position of the end effector after simulation
    """
    import mujoco
    
    # Generate random morphology
    logger.info("Generating random morphology for forward kinematics test")
    morphology = Morphology()
    
    # Print the structure for debugging
    logger.info("Morphology structure:")
    print_link_structure(morphology.base)
    
    # Get the last link (end effector)
    links_chain = morphology._get_links_chain(morphology.base)
    end_effector = links_chain[-1]
    logger.info(f"End effector: {end_effector.name}")
    
    # Convert to MJCF
    logger.info("Converting to MJCF")
    mjcf_str = morphology.to_mjcf(enable_gravity=False)
    
    # Create MuJoCo model
    logger.info("Creating MuJoCo model")
    model = mujoco.MjModel.from_xml_string(mjcf_str)
    data = mujoco.MjData(model)
    
    # Actuators should already be in the model from to_mjcf()
    logger.info(f"Model has {model.nu} actuators")
    
    # Set small random control values
    logger.info(f"Setting small random control values (magnitude: {control_magnitude}) for {model.nu} actuators")
    for i in range(model.nu):
        data.ctrl[i] = np.random.uniform(-control_magnitude, control_magnitude)
    
    # Run simulation for specified number of timesteps
    logger.info(f"Running simulation for {num_timesteps} timesteps")
    for _ in range(num_timesteps):
        mujoco.mj_step(model, data)
    
    # Get the end effector position
    # Find the body ID for the end effector
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector.name)
    end_effector_pos = data.xpos[end_effector_id].copy()
    
    logger.info(f"End effector position after simulation: {end_effector_pos}")
    
    return end_effector_pos

def test_vis():
    """Generate a random morphology and visualize it in MuJoCo."""
    # Generate random morphology
    logger.info("Generating random morphology")
    morphology = Morphology()
    
    # Print the structure for debugging
    logger.info("Morphology structure:")
    print_link_structure(morphology.base)
    
    # Convert to MJCF
    logger.info("Converting to MJCF")
    mjcf_str = morphology.to_mjcf(enable_gravity=False)
    
    # Create MuJoCo model
    logger.info("Creating MuJoCo model")
    model = mujoco.MjModel.from_xml_string(mjcf_str)
    data = mujoco.MjData(model)
    
    # Initialize control values for joints
    if model.nu > 0:  # Check if there are any actuators/controls
        logger.info(f"Setting initial control values for {model.nu} joints")
        # Set small sinusoidal control values based on joint index
        for i in range(model.nu):
            # Different frequencies and phases for varied movement
            data.ctrl[i] = 0.5 * np.sin(i * 0.7)
    else:
        logger.info(f"Setting initial control values for {model.nu} joints")
        for i in range(model.nu):
            data.ctrl[i] = 0.5 * np.sin(i * 0.7)

    links_chain = morphology._get_links_chain(morphology.base)
    end_effector = links_chain[-1]
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector.name)
    
    # Visualize
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        # Set camera position for better view
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        
        # Run simulation for a few seconds
        sim_time = 0.0
        while viewer.is_running():
            step_start = data.time
            
            # Update control values over time for continuous movement
            for i in range(model.nu):
                # Create oscillating control values with different frequencies
                data.ctrl[i] = 0.5 * np.sin(sim_time * 2.0 + i * 0.7)
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Update simulation time
            sim_time += model.opt.timestep

            # Print positions of all links in the chain
            print("\nLink tip positions:")
            for i, link in enumerate(links_chain):
                if link.name == "base":
                    continue  # Skip the base link
                
                # Calculate tip position using the mujoco_fk function
                calculated_tip_pos = mujoco_fk(data, model, link)
                
                # For all links except the last one, validate against next link's root
                if i < len(links_chain) - 1 and links_chain[i+1].name != "base":
                    # Get next link's root position
                    next_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, links_chain[i+1].name)
                    next_link_root_pos = data.xpos[next_link_id].copy()
                    
                    # Calculate distance between calculated tip and next link's root
                    distance = np.linalg.norm(calculated_tip_pos - next_link_root_pos)
                    
                    print(f"{link.name} tip (calculated): {calculated_tip_pos}")
                    print(f"{link.name} tip (next link root): {next_link_root_pos}")
                    print(f"{link.name} validation distance: {distance:.6f}")
                else:
                    # For the last link, just print the calculated tip position
                    print(f"{link.name} tip: {calculated_tip_pos}")

            import time
            time.sleep(0.2)
            
            # Break after 10 seconds
            if data.time > 10.0:
                logger.info("Exit success")
                break


if __name__ == "__main__":
    test_vis()
    # print(test_fk())
