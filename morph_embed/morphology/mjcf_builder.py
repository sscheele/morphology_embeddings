import numpy as np
from lxml import etree
from typing import List, Tuple, Optional, Dict
from morph_embed.setup_logger import logger

class MJCFBuilder:
    """Utility class for building MJCF models for morphologies."""
    
    @staticmethod
    def create_mjcf_root(model_name: str) -> etree.Element:
        """Create the root MJCF element with the given model name."""
        return etree.Element("mujoco", model=model_name)
    
    @staticmethod
    def add_compiler(mjcf_root: etree.Element) -> etree.Element:
        """Add compiler settings to the MJCF model."""
        return etree.SubElement(mjcf_root, "compiler", angle="radian", coordinate="local")
    
    @staticmethod
    def add_option(mjcf_root: etree.Element, enable_gravity: bool = False) -> etree.Element:
        """Add option element with gravity settings to the MJCF model."""
        option_elem = etree.SubElement(mjcf_root, "option")
        if enable_gravity:
            # Default MuJoCo gravity (pointing downward along z-axis)
            option_elem.set("gravity", "0 0 -9.81")
        else:
            # Zero gravity
            option_elem.set("gravity", "0 0 0")
        return option_elem
    
    @staticmethod
    def add_worldbody(mjcf_root: etree.Element) -> etree.Element:
        """Add worldbody element to the MJCF model."""
        return etree.SubElement(mjcf_root, "worldbody")
    
    @staticmethod
    def add_link_to_mujoco(parent_elem, link, prefix: str = "", color_idx: int = 0, 
                          rotation: Optional[List[float]] = None, 
                          translation: Optional[List[float]] = None):
        """
        Add a link to the MuJoCo model with the given prefix and optional transform.
        
        Args:
            parent_elem: The parent XML element
            link: The link to add
            prefix: A prefix to add to the link and joint names
            color_idx: The color index for visualization
            rotation: Optional quaternion rotation to apply to the link
            translation: Optional translation vector to apply to the link
        """
        if link.name == "base":
            # Base link is attached to the worldbody
            body = parent_elem
            
            # If this has a transform, apply it to the base
            if rotation is not None or translation is not None:
                # Use provided rotation or default
                quat = rotation if rotation is not None else [0.0, 0.0, 0.0, 1.0]
                
                # Use provided translation or default
                pos = translation if translation is not None else [0.0, 0.0, 0.0]
                
                # Create a new body for the transformed base
                body = etree.SubElement(
                    parent_elem,
                    "body",
                    name=f"{prefix}{link.name}",
                    pos=" ".join(map(str, pos)),
                    quat=" ".join(map(str, quat))
                )
        else:
            # Create a new body for this link
            pos = [0, 0, 0]  # Default position relative to parent
            
            # If this is not the first link after base, position it at the end of the parent link
            if parent_elem.tag != "worldbody":
                geom_elem = parent_elem.find(".//geom")
                if geom_elem is not None and geom_elem.get("fromto") is not None:
                    pos = [0, 0, float(geom_elem.get("fromto").split()[-1])]
            
            # Create body
            body = etree.SubElement(
                parent_elem,
                "body",
                name=f"{prefix}{link.name}",
                pos=" ".join(map(str, pos))
            )
            
            # Calculate mass based on link dimensions (assuming cylindrical shape)
            # Use a reasonable density (1000 kg/m³, similar to water)
            density = 1000.0
            if link.length > 0 and link.radius > 0:
                # Volume of a cylinder: π * r² * h
                volume = np.pi * (link.radius ** 2) * link.length
                mass = density * volume
            else:
                # Default mass for links with no dimensions
                mass = 1.0
            
            # For a cylinder along z-axis, the inertia tensor is:
            # Ixx = Iyy = m/12 * (3r² + h²)
            # Izz = m/2 * r²
            if link.length > 0 and link.radius > 0:
                ixx = iyy = mass/12.0 * (3.0 * link.radius**2 + link.length**2)
                izz = mass/2.0 * link.radius**2
            else:
                # Default inertia for links with no dimensions
                ixx = iyy = izz = 0.1
            
            # Add inertial element
            inertial = etree.SubElement(body, "inertial")
            inertial.set("pos", "0 0 0")  # Center of mass at origin
            inertial.set("mass", str(mass))
            inertial.set("diaginertia", f"{ixx} {iyy} {izz}")
            
            # Add joint at the body's origin
            if link.joint:
                if hasattr(link.joint, 'axis') and hasattr(link.joint, 'range'):
                    # This is a HingeJoint
                    axis_str = " ".join(map(str, link.joint.axis))
                    etree.SubElement(
                        body,
                        "joint",
                        name=f"{prefix}{link.joint.name}",
                        type="hinge",
                        axis=axis_str,
                        limited="true",
                        range=f"{link.joint.range[0]} {link.joint.range[1]}",
                        damping="0.2"
                    )
                elif hasattr(link.joint, 'translation') and hasattr(link.joint, 'rotation'):
                    # This is a FixedJoint
                    # For FixedJoint, we don't add a joint element but instead set the position and orientation
                    # of the body element based on the translation and rotation properties
                    
                    # Update the body position with the translation
                    trans_str = " ".join(map(str, link.joint.translation))
                    body.set("pos", trans_str)
                    
                    # Update the body orientation with the rotation (quaternion)
                    rot_str = " ".join(map(str, link.joint.rotation))
                    body.set("quat", rot_str)
                    
                    logger.info(f"Added FixedJoint for {prefix}{link.name} with translation {trans_str} and rotation {rot_str}")
            
            # Add geometry for this link
            if link.length > 0:
                # Define ROYGBV colors
                colors = [
                    "1 0 0 1",  # Red
                    "1 0.5 0 1",  # Orange
                    "1 1 0 1",  # Yellow
                    "0 1 0 1",  # Green
                    "0 0 1 1",  # Blue
                    "0.5 0 0.5 1"  # Violet
                ]
                color = colors[color_idx % len(colors)]
                
                # The capsule extends from the joint (0,0,0) along the z-axis
                fromto_str = f"0 0 0 0 0 {link.length}"
                
                logger.debug(f"  Creating capsule for {prefix}{link.name}:")
                logger.debug(f"    fromto: {fromto_str}")
                logger.debug(f"    radius: {link.radius}")
                etree.SubElement(
                    body,
                    "geom",
                    type="capsule",
                    fromto=fromto_str,
                    size=str(link.radius),
                    rgba=color
                )
        
        # Add child if it exists
        if len(link.children) > 0:
            for c in link.children:
                MJCFBuilder.add_link_to_mujoco(body, c, prefix, color_idx=color_idx + 1, 
                                            rotation=None, translation=None)
        
        return body
    
    @staticmethod
    def add_actuators(mjcf_root: etree.Element, links_chain: List, prefix: str = ""):
        """Add actuators for all joints in the links chain with the given prefix."""
        actuator_elem = etree.SubElement(mjcf_root, "actuator")
        
        for link in links_chain[1:]:  # Skip the base link
            if link.joint and hasattr(link.joint, 'axis') and hasattr(link.joint, 'range'):
                # Only add actuators for HingeJoint, not for FixedJoint
                logger.info(f"Adding actuator for joint: {prefix}{link.joint.name}")
                etree.SubElement(
                    actuator_elem,
                    "motor",
                    name=f"motor_{prefix}{link.joint.name}",
                    joint=f"{prefix}{link.joint.name}",
                    gear="1"
                )
        
        return actuator_elem
    
    @staticmethod
    def build_mjcf(links_chain, enable_gravity: bool = False, model_name: str = "morphology", prefix: str = ""):
        """
        Build a complete MJCF model for the given links chain.
        
        Args:
            links_chain: List of links in the chain
            enable_gravity: Whether to enable gravity
            model_name: Name of the model
            prefix: Prefix for link and joint names
            
        Returns:
            str: The MJCF XML string
        """
        # Create MJCF structure
        mjcf_root = MJCFBuilder.create_mjcf_root(model_name)
        MJCFBuilder.add_compiler(mjcf_root)
        MJCFBuilder.add_option(mjcf_root, enable_gravity)
        worldbody = MJCFBuilder.add_worldbody(mjcf_root)
        
        # Add the morphology tree
        MJCFBuilder.add_link_to_mujoco(worldbody, links_chain[0], prefix=prefix)
        
        # Add actuators
        MJCFBuilder.add_actuators(mjcf_root, links_chain, prefix=prefix)
        
        return etree.tostring(mjcf_root, pretty_print=True).decode("utf-8")

    @staticmethod
    def build_bimanual_mjcf(arm_morphology, rotation, translation, enable_gravity: bool = False):
        """
        Build a complete MJCF model for a bimanual morphology.
        
        Args:
            arm_morphology: The morphology of each arm
            rotation: Rotation of the second arm
            translation: Translation of the second arm
            enable_gravity: Whether to enable gravity
            
        Returns:
            str: The MJCF XML string
        """
        # Create MJCF structure
        mjcf_root = MJCFBuilder.create_mjcf_root("bimanual_morphology")
        MJCFBuilder.add_compiler(mjcf_root)
        MJCFBuilder.add_option(mjcf_root, enable_gravity)
        worldbody = MJCFBuilder.add_worldbody(mjcf_root)
        
        # Get the links chain
        links_chain = arm_morphology._get_links_chain(arm_morphology.base)
        
        # Add the first arm
        logger.info("Adding first arm to MJCF model")
        MJCFBuilder.add_link_to_mujoco(worldbody, arm_morphology.base, prefix="arm1_")
        
        # Add the second arm with rotation and translation
        logger.info(f"Adding second arm to MJCF model with rotation: {rotation}, translation: {translation}")
        MJCFBuilder.add_link_to_mujoco(
            worldbody, 
            arm_morphology.base, 
            prefix="arm2_", 
            rotation=rotation, 
            translation=translation
        )
        
        # Add actuators for both arms
        actuator_elem = etree.SubElement(mjcf_root, "actuator")
        
        # Add actuators for both arms
        for prefix in ["arm1_", "arm2_"]:
            for link in links_chain[1:]:
                if link.joint:
                    etree.SubElement(
                        actuator_elem,
                        "motor",
                        name=f"motor_{prefix}{link.joint.name}",
                        joint=f"{prefix}{link.joint.name}",
                        gear="1"
                    )
        
        return etree.tostring(mjcf_root, pretty_print=True).decode("utf-8")