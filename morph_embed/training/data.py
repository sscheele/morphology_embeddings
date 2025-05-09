import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Callable
from enum import Enum
from morph_embed.morphology import Morphology, BimanualMorphology
from morph_embed.setup_logger import logger
import mujoco

class TaskType(Enum):
    AUTOENCODER = "autoencoder"
    PHYSICS = "physics"
    MULTI_TASK = "multi_task"

class MorphologyDataset(Dataset):
    """Dataset for morphology data generation."""
    
    def __init__(
        self,
        task_type: TaskType,
        num_samples: int = 1000,
        simulation_steps: int = 10,
        transform: Optional[Callable] = None,
        bimanual_prob: float = 0.3,
        cache_size: int = 100
    ):
        """
        Initialize the dataset.
        
        Args:
            task_type: Type of task (autoencoder, physics, or multi-task)
            num_samples: Number of samples to generate
            simulation_steps: Number of simulation steps for each morphology
            transform: Optional transform to apply to the data
            bimanual_prob: Probability of generating a bimanual morphology
            cache_size: Size of the cache for generated samples
        """
        self.task_type = task_type
        self.num_samples = num_samples
        self.simulation_steps = simulation_steps
        self.transform = transform
        self.bimanual_prob = bimanual_prob
        self.cache_size = cache_size
        self.cache = {}  # Simple cache for recently generated samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Generate a sample on-the-fly or retrieve from cache.
        
        Returns a dictionary with:
            - morphology: The morphology object
            - x: Initial state
            - x_prime: Time derivative of x
            - x2: Result of MuJoCo dynamics
            - task_type: Type of task
        """
        if idx in self.cache:
            return self.cache[idx]
        
        # Generate new sample
        sample = self._generate_sample(idx)
        
        # Update cache (simple LRU strategy)
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = sample
        return sample
    
    def _generate_sample(self, idx: int) -> Dict:
        """Generate a new sample with morphology and dynamics."""
        # Determine if this should be a bimanual morphology
        is_bimanual = np.random.random() < self.bimanual_prob
        
        # Generate morphology
        if is_bimanual:
            arm_morphology = Morphology()
            morphology = BimanualMorphology(arm_morphology=arm_morphology)
        else:
            morphology = Morphology()
        
        # Convert to MJCF and create MuJoCo model
        mjcf_str = morphology.to_mjcf(enable_gravity=True)
        model = mujoco.MjModel.from_xml_string(mjcf_str)
        data = mujoco.MjData(model)
        
        # Set random initial state
        for i in range(model.nq):
            data.qpos[i] = np.random.uniform(-0.5, 0.5)
        for i in range(model.nv):
            data.qvel[i] = np.random.uniform(-0.5, 0.5)
        
        # Get initial state
        mujoco.mj_forward(model, data)
        x = np.concatenate([data.qpos, data.qvel])
        
        # Compute time derivative (velocity)
        x_prime = np.concatenate([data.qvel, data.qacc])
        
        # Step simulation to get next state
        for _ in range(self.simulation_steps):
            mujoco.mj_step(model, data)
        
        # Get result of dynamics
        x2 = np.concatenate([data.qpos, data.qvel])
        
        # Create sample dictionary
        sample = {
            "morphology": morphology,
            "x": torch.tensor(x, dtype=torch.float32),
            "x_prime": torch.tensor(x_prime, dtype=torch.float32),
            "x2": torch.tensor(x2, dtype=torch.float32),
            "task_type": self.task_type
        }
        
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class MorphologyDataModule(pl.LightningDataModule):
    """Base data module for morphology embeddings."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        simulation_steps: int = 10,
        bimanual_prob: float = 0.3,
        train_samples: int = 10000,
        val_samples: int = 1000,
        test_samples: int = 1000,
        cache_size: int = 100
    ):
        """
        Initialize the data module.
        
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            simulation_steps: Number of simulation steps for each morphology
            bimanual_prob: Probability of generating a bimanual morphology
            train_samples: Number of training samples
            val_samples: Number of validation samples
            test_samples: Number of test samples
            cache_size: Size of the cache for generated samples
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.simulation_steps = simulation_steps
        self.bimanual_prob = bimanual_prob
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.cache_size = cache_size
        
        self.task_type = None  # To be set by subclasses
    
    def prepare_data(self):
        """Nothing to download or prepare in advance."""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.train_samples)
            self.val_dataset = self._create_dataset(self.val_samples)
        
        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset(self.test_samples)
    
    def _create_dataset(self, num_samples: int) -> MorphologyDataset:
        """Create a dataset with the appropriate task type."""
        if self.task_type is None:
            raise ValueError("task_type must be set by subclasses")
        
        return MorphologyDataset(
            task_type=self.task_type,
            num_samples=num_samples,
            simulation_steps=self.simulation_steps,
            transform=self.transform if hasattr(self, "transform") else None,
            bimanual_prob=self.bimanual_prob,
            cache_size=self.cache_size
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

class AutoencoderDataModule(MorphologyDataModule):
    """Data module focused on the autoencoder reconstruction task."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type = TaskType.AUTOENCODER
        
        # You could add autoencoder-specific transforms here
        self.transform = None

class PhysicsDataModule(MorphologyDataModule):
    """Data module focused on the physics prediction task."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_type = TaskType.PHYSICS
        
        # You could add physics-specific transforms here
        self.transform = None

class MultiTaskDataModule(MorphologyDataModule):
    """Data module that handles both autoencoder and physics tasks."""
    
    def __init__(self, task_weights: Dict[str, float] = None, **kwargs):
        """
        Initialize the multi-task data module.
        
        Args:
            task_weights: Dictionary mapping task names to weights
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.task_type = TaskType.MULTI_TASK
        
        # Default to equal weighting if not specified
        self.task_weights = task_weights or {
            "autoencoder": 0.5,
            "physics": 0.5
        }
        
        # Normalize weights
        total = sum(self.task_weights.values())
        self.task_weights = {k: v / total for k, v in self.task_weights.items()}
        
        logger.info(f"MultiTaskDataModule initialized with weights: {self.task_weights}")