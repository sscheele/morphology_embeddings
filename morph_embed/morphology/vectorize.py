import torch
from abc import ABC, abstractmethod
from typing import Union, Optional

from morph_embed.setup_logger import logger
from morph_embed.morphology.base import Morphology
from morph_embed.morphology.bimanual import BimanualMorphology


class VectorizedMorphology(ABC):
    """
    Abstract base class for vectorized representations of morphologies.
    
    A VectorizedMorphology wraps either a Morphology or BimanualMorphology
    and provides methods to convert it to a tensor representation and generate
    masks for constrained attention mechanisms.
    
    Attributes:
        morphology: The underlying morphology object (Morphology or BimanualMorphology)
    """
    
    def __init__(self, morphology: Union[Morphology, BimanualMorphology]):
        """
        Initialize a VectorizedMorphology with the given morphology.
        
        Args:
            morphology: A Morphology or BimanualMorphology instance to be vectorized
        """
        if not isinstance(morphology, (Morphology, BimanualMorphology)):
            raise TypeError("morphology must be an instance of Morphology or BimanualMorphology")
        
        self.morphology = morphology
        logger.info(f"Initialized VectorizedMorphology with {type(morphology).__name__}")
    
    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the size of the vectorized representation.
        
        Returns:
            int: The length of a single morphology vector for this instance
        """
        pass
    
    @abstractmethod
    def vectorize(self) -> torch.Tensor:
        """
        Convert the morphology to a tensor representation.
        
        Returns:
            torch.Tensor: A tensor representation of the morphology
        """
        pass
    
    @abstractmethod
    def ancestor_mask(self, joint_index: int) -> torch.Tensor:
        """
        Generate a mask indicating dimensions associated with ancestors of the specified joint.
        
        This mask is used in constrained attention mechanisms where each joint may only
        attend to its ancestors in the kinematic tree.
        
        Args:
            joint_index: The index of the joint
            
        Returns:
            torch.Tensor: An int-valued mask (1 for dimensions associated with ancestors,
                         0 otherwise)
        """
        pass
    
    @abstractmethod
    def parent_mask(self, joint_index: int) -> torch.Tensor:
        """
        Generate a mask indicating dimensions associated with the parent of the specified joint.
        
        Args:
            joint_index: The index of the joint
            
        Returns:
            torch.Tensor: An int-valued mask (1 for dimensions associated with the parent,
                         0 otherwise)
        """
        pass


import numpy as np
import math


class DefaultMorphologyVectorization(VectorizedMorphology):
    """
    Default implementation of morphology vectorization.
    
    Each link in the morphology corresponds to a fixed number of dimensions in the vectorization:
    1. Offset of link base in parent's base frame (3D)
    2. Axis of rotation (3D)
    3. Positional encoding of parent joint (k-dimensional, default k=2)
    4. End-effector flag (1D)
    5. Joint type (one-hot, 2D for now, fixed or rotational)
    
    For bimanual robots, a root point is added halfway between the two arms with
    fixed joints connecting to the base of each arm.
    """
    
    def __init__(self, morphology: Union[Morphology, BimanualMorphology], positional_encoding_dim: int = 2):
        """
        Initialize a DefaultMorphologyVectorization with the given morphology.
        
        Args:
            morphology: A Morphology or BimanualMorphology instance to be vectorized
            positional_encoding_dim: Dimension of the positional encoding (default: 2)
        """
        super().__init__(morphology)
        self.positional_encoding_dim = positional_encoding_dim
        
        # Calculate dimensions per link
        self.dims_per_link = 3 + 3 + positional_encoding_dim + 1 + 2  # offset + axis + pos_encoding + ee_flag + joint_type
        
        # Get the links and build the parent-child relationships
        if isinstance(morphology, Morphology):
            self.links = morphology._get_links_chain(morphology.base)
            self.is_bimanual = False
        else:  # BimanualMorphology
            # For bimanual, we create a virtual root and connect both arms to it
            self.links = self._create_bimanual_links_chain(morphology)
            self.is_bimanual = True
        
        # Build parent indices mapping for each link
        self.parent_indices = self._build_parent_indices()
        
        # Build ancestor indices mapping for each link
        self.ancestor_indices = self._build_ancestor_indices()
        
        logger.info(f"Initialized DefaultMorphologyVectorization with {len(self.links)} links, "
                   f"{self.dims_per_link} dimensions per link")
    
    def _create_bimanual_links_chain(self, bimanual_morphology: BimanualMorphology) -> List[Link]:
        """
        Create a unified links chain for a bimanual morphology.
        
        This adds a virtual root link at the midpoint between the two arms,
        and connects both arms to it with fixed joints.
        
        Args:
            bimanual_morphology: The bimanual morphology
            
        Returns:
            List[Link]: A list of all links in the unified chain
        """
        # Create a virtual root link
        root_link = Link(
            name="virtual_root",
            length=0.0,
            radius=0.0
        )
        
        # Get the arm links
        arm_links = bimanual_morphology.arm_morphology._get_links_chain(bimanual_morphology.arm_morphology.base)
        
        # Create fixed joints to connect the arms to the root
        # First arm
        first_arm_joint = Joint(
            name="arm1_joint",
            axis=np.array([0.0, 0.0, 1.0]),  # Arbitrary axis for fixed joint
            type="fixed"
        )
        
        # Clone the first arm's base link and connect it to the root
        first_arm_base = Link(
            name="arm1_base",
            length=arm_links[0].length,
            radius=arm_links[0].radius,
            joint=first_arm_joint
        )
        
        # Connect the rest of the first arm
        current_link = first_arm_base
        for i, link in enumerate(arm_links[1:], 1):
            new_link = Link(
                name=f"arm1_{link.name}",
                length=link.length,
                radius=link.radius,
                joint=link.joint
            )
            current_link.child = new_link
            current_link = new_link
        
        # Second arm
        second_arm_joint = Joint(
            name="arm2_joint",
            axis=np.array([0.0, 0.0, 1.0]),  # Arbitrary axis for fixed joint
            type="fixed"
        )
        
        # Clone the second arm's base link and connect it to the root
        second_arm_base = Link(
            name="arm2_base",
            length=arm_links[0].length,
            radius=arm_links[0].radius,
            joint=second_arm_joint
        )
        
        # Connect the rest of the second arm
        current_link = second_arm_base
        for i, link in enumerate(arm_links[1:], 1):
            new_link = Link(
                name=f"arm2_{link.name}",
                length=link.length,
                radius=link.radius,
                joint=link.joint
            )
            current_link.child = new_link
            current_link = new_link
        
        # Connect both arms to the root
        root_link.child = first_arm_base
        
        # Find the last link of the first arm
        last_link = first_arm_base
        while last_link.child:
            last_link = last_link.child
        
        # Connect the second arm after the first arm
        last_link.child = second_arm_base
        
        # Build the complete chain
        return [root_link] + self._get_all_children(root_link)
    
    def _get_all_children(self, link: Link) -> List[Link]:
        """
        Get all children of a link recursively.
        
        Args:
            link: The starting link
            
        Returns:
            List[Link]: A list of all child links
        """
        result = []
        current = link.child
        
        while current:
            result.append(current)
            current = current.child
            
        return result
    
    def _build_parent_indices(self) -> Dict[int, int]:
        """
        Build a mapping from link index to parent link index.
        
        Returns:
            Dict[int, int]: A dictionary mapping link indices to parent link indices
        """
        parent_indices = {}
        
        for i, link in enumerate(self.links):
            if i == 0:  # Root has no parent
                parent_indices[i] = -1
            else:
                # Find the parent link
                for j, potential_parent in enumerate(self.links):
                    if potential_parent.child and potential_parent.child.name == link.name:
                        parent_indices[i] = j
                        break
        
        return parent_indices
    
    def _build_ancestor_indices(self) -> Dict[int, List[int]]:
        """
        Build a mapping from link index to a list of ancestor link indices.
        
        Returns:
            Dict[int, List[int]]: A dictionary mapping link indices to lists of ancestor link indices
        """
        ancestor_indices = {}
        
        for i in range(len(self.links)):
            ancestors = []
            current_idx = i
            
            while self.parent_indices[current_idx] != -1:
                parent_idx = self.parent_indices[current_idx]
                ancestors.append(parent_idx)
                current_idx = parent_idx
            
            ancestor_indices[i] = ancestors
        
        return ancestor_indices
    
    def _positional_encoding(self, pos: int) -> np.ndarray:
        """
        Generate a positional encoding for a given position.
        
        Args:
            pos: The position to encode
            
        Returns:
            np.ndarray: The positional encoding
        """
        encoding = np.zeros(self.positional_encoding_dim)
        
        if self.positional_encoding_dim >= 1:
            encoding[0] = math.sin(pos)
        
        if self.positional_encoding_dim >= 2:
            encoding[1] = math.cos(pos)
        
        # For higher dimensions, use the standard transformer positional encoding formula
        if self.positional_encoding_dim > 2:
            for i in range(2, self.positional_encoding_dim):
                if i % 2 == 0:
                    encoding[i] = math.sin(pos / (10000 ** (i / self.positional_encoding_dim)))
                else:
                    encoding[i] = math.cos(pos / (10000 ** ((i - 1) / self.positional_encoding_dim)))
        
        return encoding
    
    @property
    def size(self) -> int:
        """
        Get the size of the vectorized representation.
        
        Returns:
            int: The length of a single morphology vector for this instance
        """
        return len(self.links) * self.dims_per_link
    
    def vectorize(self) -> torch.Tensor:
        """
        Convert the morphology to a tensor representation.
        
        Returns:
            torch.Tensor: A tensor representation of the morphology
        """
        # Initialize the vector
        vector = np.zeros(self.size)
        
        for i, link in enumerate(self.links):
            start_idx = i * self.dims_per_link
            
            # 1. Offset of link base in parent's base frame (3D)
            if i > 0:
                parent_idx = self.parent_indices[i]
                parent_link = self.links[parent_idx]
                # In this project, the offset is [0, 0, parent_length]
                vector[start_idx:start_idx+3] = np.array([0.0, 0.0, parent_link.length])
            
            # 2. Axis of rotation (3D)
            if link.joint:
                vector[start_idx+3:start_idx+6] = link.joint.axis
            
            # 3. Positional encoding of parent joint (k-dimensional)
            pos_encoding = self._positional_encoding(i)
            vector[start_idx+6:start_idx+6+self.positional_encoding_dim] = pos_encoding
            
            # 4. End-effector flag (1D)
            if link.child is None:
                vector[start_idx+6+self.positional_encoding_dim] = 1.0
            
            # 5. Joint type (one-hot, 2D)
            if link.joint:
                if link.joint.type == "fixed":
                    vector[start_idx+6+self.positional_encoding_dim+1] = 1.0
                else:  # "hinge" or other rotational joint
                    vector[start_idx+6+self.positional_encoding_dim+2] = 1.0
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def ancestor_mask(self, joint_index: int) -> torch.Tensor:
        """
        Generate a mask indicating dimensions associated with ancestors of the specified joint.
        
        Args:
            joint_index: The index of the joint
            
        Returns:
            torch.Tensor: An int-valued mask (1 for dimensions associated with ancestors,
                         0 otherwise)
        """
        mask = np.zeros(self.size, dtype=int)
        
        # Include the joint itself
        start_idx = joint_index * self.dims_per_link
        end_idx = start_idx + self.dims_per_link
        mask[start_idx:end_idx] = 1
        
        # Include all ancestors
        for ancestor_idx in self.ancestor_indices.get(joint_index, []):
            start_idx = ancestor_idx * self.dims_per_link
            end_idx = start_idx + self.dims_per_link
            mask[start_idx:end_idx] = 1
        
        return torch.tensor(mask, dtype=torch.int)
    
    def parent_mask(self, joint_index: int) -> torch.Tensor:
        """
        Generate a mask indicating dimensions associated with the parent of the specified joint.
        
        Args:
            joint_index: The index of the joint
            
        Returns:
            torch.Tensor: An int-valued mask (1 for dimensions associated with the parent,
                         0 otherwise)
        """
        mask = np.zeros(self.size, dtype=int)
        
        parent_idx = self.parent_indices.get(joint_index, -1)
        if parent_idx >= 0:
            start_idx = parent_idx * self.dims_per_link
            end_idx = start_idx + self.dims_per_link
            mask[start_idx:end_idx] = 1
        
        return torch.tensor(mask, dtype=torch.int)
