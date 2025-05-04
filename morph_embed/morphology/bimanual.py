import numpy as np
from lxml import etree
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from morph_embed.setup_logger import logger

import mujoco
import mujoco.viewer

from morph_embed.morphology.base import Link, Joint, Morphology, random_unit_vector, mujoco_fk

class BimanualMorphology:
    """
    A class representing a bimanual morphology with two symmetrical arms.
    
    The bimanual morphology consists of two identical arms (represented by the same
    Morphology object) with a transform that specifies how the arms are oriented
    relative to one another.
    """
    
    def __init__(self, arm_morphology: Optional[Morphology] = None,
                 rotation: Optional[np.ndarray] = None,
                 translation: Optional[np.ndarray] = None):
        """
        Initialize a BimanualMorphology with the given arm morphology and transform.
        
        Args:
            arm_morphology: The morphology of each arm (both arms are identical)
            rotation: A quaternion [x, y, z, w] representing the rotation of the second arm
                      relative to the first. If None, a default rotation is used.
            translation: A 3D vector [x, y, z] representing the translation of the second arm
                         relative to the first. If None, a default translation is used.
        """
        self.logger = logger
        
        # Create or use the provided arm morphology
        if arm_morphology is None:
            self.logger.info("Generating random arm morphology for bimanual system")
            self.arm_morphology = Morphology()
        else:
            self.arm_morphology = arm_morphology
            
        # Set the rotation between arms
        if rotation is None:
            # Default rotation: None
            self.rotation = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        else:
            self.rotation = np.array(rotation)
        
        # Set the translation between arms
        if translation is None:
            # Default translation: random coplanar between 0 and 0.75*sum(joint_lengths)
            def sum_joint_lens(link: Link):
                if link.child is None:
                    return link.length
                return link.length + sum_joint_lens(link.child)
            total_len = sum_joint_lens(self.arm_morphology.base)
            max_translation = 0.75*total_len
            min_translation = 0.2*total_len
            translation_vec = np.random.uniform(min_translation, max_translation)*random_unit_vector(size=2)
            
            self.translation = np.array(list(translation_vec) + [0.0])  # [x, y, z]
        else:
            self.translation = np.array(translation)
            
        self.logger.info(f"BimanualMorphology initialized with rotation: {self.rotation}, translation: {self.translation}")
    
    def to_mjcf(self, enable_gravity: bool = False) -> str:
        """
        Convert the bimanual morphology to MJCF format string.
        
        Args:
            enable_gravity: Whether to enable gravity in the simulation (default: False)
            
        Returns:
            str: The MJCF XML string representing the bimanual morphology
        """
        from morph_embed.morphology.mjcf_builder import MJCFBuilder
        
        return MJCFBuilder.build_bimanual_mjcf(
            arm_morphology=self.arm_morphology,
            rotation=self.rotation,
            translation=self.translation,
            enable_gravity=enable_gravity
        )
    
    def to_dict(self) -> Dict:
        """
        Convert the bimanual morphology to a dictionary representation.
        
        Returns:
            Dict: A dictionary representation of the bimanual morphology
        """
        return {
            'arm_morphology': self.arm_morphology.to_dict(),
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BimanualMorphology':
        """
        Create a BimanualMorphology from a dictionary representation.
        
        Args:
            data: Dictionary containing the bimanual morphology data
            
        Returns:
            BimanualMorphology: A new BimanualMorphology instance
        """
        arm_morphology = Morphology(params=data['arm_morphology'])
        rotation = np.array(data['rotation']) if 'rotation' in data else None
        translation = np.array(data['translation']) if 'translation' in data else None
        
        return cls(arm_morphology=arm_morphology, rotation=rotation, translation=translation)
    
    def get_end_effector_positions(self, model, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the positions of both end effectors.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The 3D positions of both end effectors
        """
        # Get the links chain from the arm morphology
        links_chain = self.arm_morphology._get_links_chain(self.arm_morphology.base)
        end_effector = links_chain[-1]

        mock_ee1_link = Link(f"arm1_{end_effector.name}", end_effector.length, end_effector.radius)
        mock_ee2_link = Link(f"arm2_{end_effector.name}", end_effector.length, end_effector.radius)
        
        return mujoco_fk(data, model, mock_ee1_link), mujoco_fk(data, model, mock_ee2_link)

def test_bimanual_vis():
    """Generate a random bimanual morphology and visualize it in MuJoCo."""
    # Generate random bimanual morphology
    logger.info("Generating random bimanual morphology")
    # Use default rotation (180° around Y-axis) and translation (1 unit in X-axis)
    bimanual = BimanualMorphology()
    
    # Convert to MJCF
    logger.info("Converting to MJCF")
    mjcf_str = bimanual.to_mjcf()
    
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