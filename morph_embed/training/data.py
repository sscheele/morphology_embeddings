import torch
from abc import ABC, abstractmethod
from morph_embed.morphology.base import Morphology
from morph_embed.morphology.vectorize import VectorizedMorphology, DefaultMorphologyVectorization
from morph_embed.setup_logger import logger

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Callable

from typing import Optional, Dict

class MorphologyTask(ABC):
    morph: Morphology

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_sample(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

class MorphologyAutoencoderTask(MorphologyTask):
    def __init__(self,
            vectorizer: VectorizedMorphology,
            autoencoder_dim: int,
            other_task: Optional[MorphologyTask]=None):
        self.morph = vectorizer.morphology
        self.other_task = other_task
        self.vectorizer = vectorizer
        self.autoencoder_dim = autoencoder_dim
    
    @property
    def size(self) -> int:
        if self.other_task is None:
            return self.autoencoder_dim
        return self.autoencoder_dim + self.other_task.size

    def get_sample(self) -> Dict[str, torch.Tensor]:
        out = self.other_task.get_sample()

        auto_vec = self.vectorizer.vectorize()

        # pad with zeros to hit a consistent input size
        pad_len = self.autoencoder_dim - auto_vec.size(0)

        if pad_len > 0:
            auto_vec = torch.cat([auto_vec, torch.zeros(pad_len, dtype=auto_vec.dtype)])
        
        out['autoencoder_vec'] = auto_vec

class MorphologyDynamicsTask(MorphologyTask):
    """
    MorphologyDynamicsTask challenges the network to learn to predict the robot's future
    state given its current state and control inputs
    """
    
    def __init__(self,
                 morphology: Optional[Morphology] = None,
                 num_timesteps: int = 10,
                 control_magnitude: float = 0.1,
                 state_dim: int = 12):
        """
        Initialize a MorphologyDynamicsTask.
        
        Args:
            morphology: Optional morphology to use. If None, a random morphology will be generated
                       for each sample.
            num_timesteps: Number of simulation timesteps to run for each sample
            control_magnitude: Maximum magnitude of random control values
            state_dim: Dimension of the state vector (position + velocity)
        """
        from morph_embed.setup_logger import logger
        
        # If no morphology is provided, we'll generate a random one for each sample
        self.morph = morphology
        self.num_timesteps = num_timesteps
        self.control_magnitude = control_magnitude
        self.state_dim = state_dim
        
        logger.info(f"Initialized MorphologyDynamicsTask with num_timesteps={num_timesteps}, "
                   f"control_magnitude={control_magnitude}")
    
    @property
    def size(self) -> int:
        """
        Get the size of the state vector.
        
        Returns:
            int: The dimension of the state vector
        """
        return self.state_dim
    
    def get_sample(self) -> Dict[str, torch.Tensor]:
        """
        Generate a sample by simulating the morphology dynamics.
        
        This method:
        1. Generates a random morphology if one wasn't provided
        2. Converts it to MJCF format
        3. Creates a MuJoCo model and data
        4. Sets random control values
        5. Simulates the dynamics for a few timesteps
        6. Returns the initial state, its derivative, and the final state
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - x: Initial state tensor
                - x_prime: Time derivative of the initial state
                - x2: State after simulation (future state)
        """
        import mujoco
        import numpy as np
        from morph_embed.setup_logger import logger
        from morph_embed.morphology.base import Morphology
        
        # Generate a random morphology if one wasn't provided
        if self.morph is None:
            morphology = Morphology()
        else:
            morphology = self.morph
            
        # Convert to MJCF
        mjcf_str = morphology.to_mjcf(enable_gravity=False)
        
        # Create MuJoCo model
        model = mujoco.MjModel.from_xml_string(mjcf_str)
        data = mujoco.MjData(model)
        
        # Set random initial state
        for i in range(model.nq):
            # Random joint positions within joint limits
            if model.jnt_limited[i]:
                data.qpos[i] = np.random.uniform(model.jnt_range[i, 0], model.jnt_range[i, 1])
            else:
                data.qpos[i] = np.random.uniform(-1.0, 1.0)
                
        starting_controls = np.random.uniform(-self.control_magnitude, self.control_magnitude, size=model.nu)
        
        # Set random control values
        for i in range(model.nu):
            data.ctrl[i] = starting_controls
        
        # Get the initial state
        mujoco.mj_forward(model, data)
        
        # Extract the initial state (position and velocity)
        x_pos = np.zeros(model.nq)
        x_vel = np.zeros(model.nv)
        
        for i in range(model.nq):
            x_pos[i] = data.qpos[i]
        
        for i in range(model.nv):
            x_vel[i] = data.qvel[i]
        
        # Combine position and velocity for the full state
        x = np.concatenate([x_pos, x_vel])
        
        # Get the time derivative (acceleration)
        x_prime = np.zeros_like(x)
        x_prime[:model.nq] = x_vel  # Position derivative is velocity
        x_prime[model.nq:] = data.qacc  # Velocity derivative is acceleration
        
        # Run simulation for specified number of timesteps
        for _ in range(self.num_timesteps):
            mujoco.mj_step(model, data)
        
        # Extract the final state
        x2_pos = np.zeros(model.nq)
        x2_vel = np.zeros(model.nv)
        
        for i in range(model.nq):
            x2_pos[i] = data.qpos[i]
        
        for i in range(model.nv):
            x2_vel[i] = data.qvel[i]
        
        # Combine position and velocity for the full final state
        x2 = np.concatenate([x2_pos, x2_vel])
        
        # Pad or truncate to match the expected state dimension
        if len(x) < self.state_dim:
            x = np.pad(x, (0, self.state_dim - len(x)))
            x_prime = np.pad(x_prime, (0, self.state_dim - len(x_prime)))
            x2 = np.pad(x2, (0, self.state_dim - len(x2)))
        elif len(x) > self.state_dim:
            x = x[:self.state_dim]
            x_prime = x_prime[:self.state_dim]
            x2 = x2[:self.state_dim]
        
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_prime_tensor = torch.tensor(x_prime, dtype=torch.float32)
        x2_tensor = torch.tensor(x2, dtype=torch.float32)
        
        return {
            "x": x_tensor,
            "x_prime": x_prime_tensor,
            "x2": x2_tensor
        }

class MorphologyDynamicsAutoencoderDataset(Dataset):
    """
    Dataset that generates samples on-the-fly with a new random morphology for each sample.
    """
    
    def __init__(self, num_samples: int, autoencoder_dim: int, state_dim: int,
                 num_timesteps: int, control_magnitude: float):
        """
        Initialize the dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            autoencoder_dim: Dimension of the autoencoder input/output
            state_dim: Dimension of the state vector
            num_timesteps: Number of simulation timesteps
            control_magnitude: Maximum magnitude of control values
        """
        self.num_samples = num_samples
        self.autoencoder_dim = autoencoder_dim
        self.state_dim = state_dim
        self.num_timesteps = num_timesteps
        self.control_magnitude = control_magnitude
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a sample with a new random morphology.
        
        Args:
            idx: Index of the sample (not used, as samples are generated randomly)
            
        Returns:
            Dict[str, torch.Tensor]: A dictionary of tensors for the sample
        """
        # Create a new random morphology for each sample
        morphology = Morphology()
        vectorizer = DefaultMorphologyVectorization(morphology)
        
        # Create dynamics task with the new morphology
        dynamics_task = MorphologyDynamicsTask(
            morphology=morphology,
            num_timesteps=self.num_timesteps,
            control_magnitude=self.control_magnitude,
            state_dim=self.state_dim
        )
        
        # Create autoencoder task with the dynamics task
        autoencoder_task = MorphologyAutoencoderTask(
            vectorizer=vectorizer,
            autoencoder_dim=self.autoencoder_dim,
            other_task=dynamics_task
        )
        
        # Get sample from the combined task
        return autoencoder_task.get_sample()

class MorphologyDynamicsAutoencoder(pl.LightningDataModule):
    """
    LightningDataModule for the morphology dynamics autoencoder task.
    
    This data module prepares dataloaders using both the MorphologyAutoencoderTask
    and MorphologyDynamicsTask classes to train a model that can both autoencode
    morphology states and predict dynamics. A new random morphology is generated
    for each sample.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        autoencoder_dim: int = 64,
        state_dim: int = 12,
        num_timesteps: int = 10,
        control_magnitude: float = 0.1,
        train_samples: int = 1000,
        val_samples: int = 200,
        test_samples: int = 200,
    ):
        """
        Initialize the MorphologyDynamicsAutoencoder data module.
        
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            autoencoder_dim: Dimension of the autoencoder input/output
            state_dim: Dimension of the state vector (position + velocity)
            num_timesteps: Number of simulation timesteps to run for each sample
            control_magnitude: Maximum magnitude of random control values
            train_samples: Number of training samples
            val_samples: Number of validation samples
            test_samples: Number of test samples
        """
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.autoencoder_dim = autoencoder_dim
        self.state_dim = state_dim
        self.num_timesteps = num_timesteps
        self.control_magnitude = control_magnitude
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        
        logger.info(f"Initialized MorphologyDynamicsAutoencoder with batch_size={batch_size}, "
                   f"autoencoder_dim={autoencoder_dim}, state_dim={state_dim}")
        
        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up the data module by creating datasets.
        
        Args:
            stage: Stage of setup ('fit', 'validate', 'test', or None)
        """
        from morph_embed.setup_logger import logger
        
        logger.info(f"Setting up MorphologyDynamicsAutoencoder for stage: {stage}")
        
        # Create datasets for each stage
        if stage == 'fit' or stage is None:
            self.train_dataset = MorphologyDynamicsAutoencoderDataset(
                num_samples=self.train_samples,
                autoencoder_dim=self.autoencoder_dim,
                state_dim=self.state_dim,
                num_timesteps=self.num_timesteps,
                control_magnitude=self.control_magnitude
            )
            
            self.val_dataset = MorphologyDynamicsAutoencoderDataset(
                num_samples=self.val_samples,
                autoencoder_dim=self.autoencoder_dim,
                state_dim=self.state_dim,
                num_timesteps=self.num_timesteps,
                control_magnitude=self.control_magnitude
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = MorphologyDynamicsAutoencoderDataset(
                num_samples=self.test_samples,
                autoencoder_dim=self.autoencoder_dim,
                state_dim=self.state_dim,
                num_timesteps=self.num_timesteps,
                control_magnitude=self.control_magnitude
            )
    
    def train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader.
        
        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Get the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Get the test dataloader.
        
        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
