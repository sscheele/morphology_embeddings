import numpy as np
from lxml import etree
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from morph_embed.setup_logger import logger

import mujoco
import mujoco.viewer

from morph_embed.morphology.base import Link, Joint, HingeJoint, FixedJoint, Morphology, random_unit_vector, random_rotation_quat, mujoco_fk, print_link_structure


class BimanualMorphology(Morphology):
    """
    A bimanual morphology class that creates a robot with two identical arms.
    
    This class extends the Morphology class and creates a 'phantom' base joint
    with two copies of the same morphology as its children. The distance between
    the base joints of the arms is a random number between 20% and 80% of the
    length of the arm, and the rotations from the root to the base of each arm
    are also random. However, the base joints of each arm have the same height (z-axis).
    """
    
    def __init__(self, arm_morphology: Optional[Morphology] = None):
        """
        Initialize a bimanual morphology with two identical arms.
        
        Args:
            arm_morphology: Optional single-arm morphology. If not provided,
                           a random morphology will be created.
        """
        # Create a base Morphology instance without random generation
        super().__init__(params={})
        
        # Create or use the provided arm morphology
        if arm_morphology is None:
            self.arm_morphology = Morphology()
        else:
            self.arm_morphology = arm_morphology
        
        # Calculate the total arm length for spacing
        arm_links = self.arm_morphology._get_links_chain(self.arm_morphology.base)
        total_arm_length = sum(link.length for link in arm_links if link.name != "base")
        
        # Calculate random distance between arms (20-80% of arm length)
        self.arm_distance = np.random.uniform(0.2 * total_arm_length, 0.8 * total_arm_length)
        logger.info(f"Creating bimanual morphology with arm distance: {self.arm_distance}")
        
        # Create the phantom base with small dimensions to ensure proper mass/inertia
        self.base = Link(
            name="bimanual_base",
            length=0.0,  # Small but non-zero length
            radius=0.0   # Small but non-zero radius
        )
        
        # # Generate random rotations for the arms
        # # We'll use random quaternions but ensure they keep the same height (z-axis)
        # # This means we'll only rotate around the z-axis
        # arm1_angle = np.random.uniform(0, 2 * np.pi)  # Random angle around z-axis
        # arm2_angle = np.random.uniform(0, 2 * np.pi)  # Random angle around z-axis
        
        # # Convert angles to quaternions (rotation around z-axis)
        # # Quaternion format: [x, y, z, w] where [x, y, z] is the axis scaled by sin(angle/2)
        # # and w is cos(angle/2)
        # arm1_quat = [0, 0, np.sin(arm1_angle/2), np.cos(arm1_angle/2)]
        # arm2_quat = [0, 0, np.sin(arm2_angle/2), np.cos(arm2_angle/2)]
        arm1_quat = [0,0,0,1]
        arm2_quat = [0,0,0,1]
        
        # logger.info(f"Arm 1 rotation: angle={arm1_angle}, quat={arm1_quat}")
        # logger.info(f"Arm 2 rotation: angle={arm2_angle}, quat={arm2_quat}")
        
        # Create the first arm base with small dimensions
        # Position it at (-arm_distance/2, 0, 0) with random rotation around z-axis
        arm1_base = Link(
            name="arm1_base",
            length=0.0,  # Small but non-zero length
            radius=0.0,  # Small but non-zero radius
            joint=FixedJoint(
                name="arm1_base_joint",
                translation=np.array([-self.arm_distance/2, 0, 0]),  # Position on negative x-axis
                rotation=np.array(arm1_quat)  # Random rotation around z-axis
            )
        )
        
        # Create the second arm base with small dimensions
        # Position it at (arm_distance/2, 0, 0) with random rotation around z-axis
        arm2_base = Link(
            name="arm2_base",
            length=0.0,  # Small but non-zero length
            radius=0.0,  # Small but non-zero radius
            joint=FixedJoint(
                name="arm2_base_joint",
                translation=np.array([self.arm_distance/2, 0, 0]),  # Position on positive x-axis
                rotation=np.array(arm2_quat)  # Random rotation around z-axis
            )
        )
        
        # Add both arm bases as children of the phantom base
        self.base.children = [arm1_base, arm2_base]
        
        # Clone the arm morphology for each arm
        # First arm
        arm1_links = self._clone_arm(self.arm_morphology.base, "arm1_")
        arm1_base.children = [arm1_links]
        
        # Second arm
        arm2_links = self._clone_arm(self.arm_morphology.base, "arm2_")
        arm2_base.children = [arm2_links]
        
        # Compute end effectors
        self._compute_end_effectors()
        
        logger.info("Bimanual morphology created successfully")
        logger.info("Morphology structure:")
        print_link_structure(self.base)
    
    def _clone_arm(self, link: Link, prefix: str) -> Link:
        """
        Clone a link and all its children with a new prefix.
        
        Args:
            link: The link to clone
            prefix: Prefix to add to all link and joint names
            
        Returns:
            Link: The cloned link
        """
        if link.name == "base":
            # Skip the base link and clone its children directly
            if len(link.children) > 0:
                return self._clone_arm(link.children[0], prefix)
            return None
        
        # Clone the current link
        new_link = Link(
            name=f"{prefix}{link.name}",
            length=link.length,
            radius=link.radius
        )
        
        # Clone the joint if it exists
        if link.joint:
            if isinstance(link.joint, HingeJoint):
                new_link.joint = HingeJoint(
                    name=f"{prefix}{link.joint.name}",
                    axis=link.joint.axis.copy(),
                    range=link.joint.range
                )
            elif isinstance(link.joint, FixedJoint):
                new_link.joint = FixedJoint(
                    name=f"{prefix}{link.joint.name}",
                    translation=link.joint.translation,
                    rotation=link.joint.rotation
                )
        
        # Clone children recursively
        if len(link.children) > 0:
            new_link.children = [self._clone_arm(child, prefix) for child in link.children]
        
        return new_link
    
    def to_mjcf(self, enable_gravity: bool = False) -> str:
        """
        Convert the bimanual morphology to MJCF format string.
        
        Args:
            enable_gravity: Whether to enable gravity in the simulation (default: False)
        
        Returns:
            str: The MJCF XML string representing the morphology
        """
        # Get the links chain
        links_chain = self._get_links_chain(self.base)
        
        # Use the builder to create the MJCF
        from morph_embed.morphology.mjcf_builder import MJCFBuilder
        return MJCFBuilder.build_mjcf(
            links_chain=links_chain,
            enable_gravity=enable_gravity,
            model_name="bimanual_morphology",
            prefix=""
        )
    
    def get_end_effector_positions(self, model, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the positions of both end effectors.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Positions of both end effectors
        """
        # Find the end effectors for each arm
        arm1_ee = None
        arm2_ee = None
        
        for ee_name in self.end_effectors:
            if ee_name.startswith("arm1_"):
                arm1_ee = ee_name
            elif ee_name.startswith("arm2_"):
                arm2_ee = ee_name
        
        if not arm1_ee or not arm2_ee:
            logger.error("Could not find end effectors for both arms")
            return np.zeros(3), np.zeros(3)
        
        # Get the links by name
        links_chain = self._get_links_chain(self.base)
        arm1_ee_link = next((link for link in links_chain if link.name == arm1_ee), None)
        arm2_ee_link = next((link for link in links_chain if link.name == arm2_ee), None)
        
        if not arm1_ee_link or not arm2_ee_link:
            logger.error("Could not find end effector links")
            return np.zeros(3), np.zeros(3)
        
        # Calculate the tip positions using forward kinematics
        arm1_pos = mujoco_fk(data, model, arm1_ee_link)
        arm2_pos = mujoco_fk(data, model, arm2_ee_link)
        
        return arm1_pos, arm2_pos


def test_bimanual_vis():
    """Generate a random bimanual morphology and visualize it in MuJoCo."""
    # Generate random bimanual morphology
    logger.info("Generating random bimanual morphology")
    # Use default rotation (180° around Y-axis) and translation (1 unit in X-axis)
    bimanual = BimanualMorphology()
    
    # Convert to MJCF
    logger.info("Converting to MJCF")
    mjcf_str = bimanual.to_mjcf()
    
    # Save MJCF to file for debugging
    # with open("bimanual_debug.xml", "w") as f:
    #     f.write(mjcf_str)
    # logger.info("Saved MJCF to bimanual_debug.xml for debugging")
    
    # Create MuJoCo model
    logger.info("Creating MuJoCo model")
    try:
        model = mujoco.MjModel.from_xml_string(mjcf_str)
        data = mujoco.MjData(model)
        logger.info("Successfully created MuJoCo model")
    except Exception as e:
        logger.error(f"Error creating MuJoCo model: {e}")
        raise
    
    # Initialize control values for joints
    if model.nu > 0:  # Check if there are any actuators/controls
        logger.info(f"Setting initial control values for {model.nu} joints")
        # Set small sinusoidal control values based on joint index
        for i in range(model.nu):
            # Different frequencies and phases for varied movement
            data.ctrl[i] = 0.3 * np.sin(i * 0.7)
    
    # Visualize
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        # Set camera position for better view
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        
        # Run simulation for a few seconds
        sim_time = 0.0
        while viewer.is_running():
            step_start = data.time
            
            # Update control values over time for continuous movement
            for i in range(model.nu):
                # Create oscillating control values with different frequencies
                data.ctrl[i] = 0.3 * np.sin(sim_time * 2.0 + i * 0.7)
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Update simulation time
            sim_time += model.opt.timestep
            
            # Get end effector positions
            arm1_pos, arm2_pos = bimanual.get_end_effector_positions(model, data)
            
            # Print end effector positions occasionally
            if int(sim_time * 10) % 10 == 0:
                logger.info(f"Arm 1 end effector: {arm1_pos}")
                logger.info(f"Arm 2 end effector: {arm2_pos}")
                logger.info(f"Distance between end effectors: {np.linalg.norm(arm1_pos - arm2_pos)}")
            
            import time
            time.sleep(0.01)
            
            # Break after 10 seconds
            if data.time > 10.0:
                logger.info("Exit success")
                break

if __name__ == "__main__":
    # test_vis()
    # print(test_fk())
    test_bimanual_vis()