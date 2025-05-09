import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from morph_embed.setup_logger import logger
from morph_embed.morphology import Morphology, BimanualMorphology
from .data import MultiTaskDataModule, TaskType

class ConstrainedAttention(nn.Module):
    """
    Attention module where each joint can only attend to its ancestors in the kinematic tree.
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        """
        Initialize the constrained attention module.
        
        Args:
            embed_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        kinematic_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the constrained attention module.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            kinematic_mask: Boolean mask of shape [batch_size, seq_len, seq_len] where True values
                           indicate that a joint can attend to another joint (its ancestor)
                           
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply kinematic mask if provided
        if kinematic_mask is not None:
            # Expand mask for multiple heads
            expanded_mask = kinematic_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output

class Encoder(nn.Module):
    """
    Encoder network that embeds morphology states using constrained attention.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 256, 
        latent_dim: int = 128,
        num_layers: int = 3
    ):
        """
        Initialize the encoder network.
        
        Args:
            input_dim: Dimension of the input (state + derivative)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of the latent space
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers with constrained attention
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                ConstrainedAttention(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final projection to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        kinematic_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            kinematic_masks: List of kinematic masks for each transformer layer
            
        Returns:
            Latent representation of shape [batch_size, latent_dim]
        """
        # Reshape input for transformer if needed
        batch_size = x.shape[0]
        seq_len = x.shape[1] // self.input_dim if len(x.shape) == 2 else x.shape[1]
        
        if len(x.shape) == 2:
            x = x.reshape(batch_size, seq_len, self.input_dim)
        
        # Initial projection
        h = self.input_proj(x)
        
        # Apply transformer layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.transformer_layers, self.layer_norms)):
            mask = None if kinematic_masks is None else kinematic_masks[i]
            residual = h
            h = layer[0](h, mask)  # Apply constrained attention
            h = layer[1](h)  # Apply layer norm
            h = residual + layer[2:](h)  # Apply FFN with residual connection
            h = norm(h)  # Final layer norm
        
        # Final projection to latent space
        z = self.output_proj(h)
        
        # Average pooling over sequence dimension if needed
        if len(z.shape) > 2:
            z = z.mean(dim=1)
            
        return z

class PhysicsNetwork(nn.Module):
    """
    Physics network that predicts the next state from the latent representation.
    """
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dim: int = 256, 
        num_layers: int = 3
    ):
        """
        Initialize the physics network.
        
        Args:
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            num_layers: Number of layers in the network
        """
        super().__init__()
        
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the physics network.
        
        Args:
            z: Latent representation of shape [batch_size, latent_dim]
            
        Returns:
            Predicted next state in latent space of shape [batch_size, latent_dim]
        """
        return self.network(z)

class Decoder(nn.Module):
    """
    Decoder network that reconstructs the state from the latent representation.
    """
    def __init__(
        self, 
        latent_dim: int, 
        hidden_dim: int = 256, 
        output_dim: int = None, 
        num_layers: int = 3
    ):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of the output (state + derivative)
            num_layers: Number of layers in the network
        """
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else latent_dim * 2
        
        layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent representation of shape [batch_size, latent_dim]
            
        Returns:
            Reconstructed state of shape [batch_size, output_dim]
        """
        return self.network(z)

class MorphologyEmbedder(pl.LightningModule):
    """
    LightningModule for training morphology embeddings.
    """
    def __init__(
        self,
        input_dim: int = 12,  # Default for simple morphologies (6 DOF position + 6 DOF velocity)
        hidden_dim: int = 256,
        latent_dim: int = 128,
        learning_rate: float = 1e-4,
        alpha: float = 0.7,  # Weight for autoencoder loss
        beta: float = 0.3,   # Weight for physics loss
        weight_decay: float = 1e-5
    ):
        """
        Initialize the morphology embedder.
        
        Args:
            input_dim: Dimension of the input state (position + velocity)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate for the optimizer
            alpha: Weight for the autoencoder loss
            beta: Weight for the physics loss
            weight_decay: Weight decay for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.weight_decay = weight_decay
        
        # Define networks
        self.encoder = Encoder(
            input_dim=input_dim * 2,  # x and x_prime concatenated
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.physics_network = PhysicsNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim * 2  # x and x_prime concatenated
        )
        
        logger.info(f"Initialized MorphologyEmbedder with latent_dim={latent_dim}, alpha={alpha}, beta={beta}")
        
    def forward(self, x: torch.Tensor, x_prime: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: State tensor of shape [batch_size, input_dim]
            x_prime: Time derivative tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - z: Latent representation
                - x_reconstructed: Reconstructed state and derivative
                - z_physics: Latent representation after physics network
                - x2_predicted: Predicted next state
        """
        # Concatenate state and derivative
        x_combined = torch.cat([x, x_prime], dim=1)
        
        # Encode
        z = self.encoder(x_combined)
        
        # Reconstruct
        x_reconstructed = self.decoder(z)
        
        # Apply physics network
        z_physics = self.physics_network(z)
        
        # Decode physics prediction
        x2_predicted = self.decoder(z_physics)
        
        return {
            "z": z,
            "x_reconstructed": x_reconstructed,
            "z_physics": z_physics,
            "x2_predicted": x2_predicted
        }
    
    def _compute_losses(
        self, 
        x: torch.Tensor, 
        x_prime: torch.Tensor, 
        x2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for the model.
        
        Args:
            x: State tensor of shape [batch_size, input_dim]
            x_prime: Time derivative tensor of shape [batch_size, input_dim]
            x2: Next state tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary containing:
                - autoencoder_loss: Reconstruction loss
                - physics_loss: Physics prediction loss
                - loss: Combined loss
        """
        # Forward pass
        outputs = self.forward(x, x_prime)
        
        # Concatenate state and derivative for autoencoder loss
        x_combined = torch.cat([x, x_prime], dim=1)
        
        # Compute autoencoder loss
        autoencoder_loss = F.mse_loss(x_combined, outputs["x_reconstructed"])
        
        # Compute physics loss (only on the state part, not the derivative)
        physics_loss = F.mse_loss(x2, outputs["x2_predicted"][:, :self.input_dim])
        
        # Compute combined loss
        loss = self.alpha * autoencoder_loss + self.beta * physics_loss
        
        return {
            "autoencoder_loss": autoencoder_loss,
            "physics_loss": physics_loss,
            "loss": loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Dictionary containing:
                - x: State tensor
                - x_prime: Time derivative tensor
                - x2: Next state tensor
                - morphology: Morphology object (not used in forward pass)
                - task_type: Task type
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        # Extract data
        x = batch["x"]
        x_prime = batch["x_prime"]
        x2 = batch["x2"]
        
        # Compute losses
        losses = self._compute_losses(x, x_prime, x2)
        
        # Log metrics
        self.log("train_loss", losses["loss"], prog_bar=True)
        self.log("train_autoencoder_loss", losses["autoencoder_loss"])
        self.log("train_physics_loss", losses["physics_loss"])
        
        logger.info(f"Batch {batch_idx}: train_loss={losses['loss']:.4f}, "
                   f"autoencoder_loss={losses['autoencoder_loss']:.4f}, "
                   f"physics_loss={losses['physics_loss']:.4f}")
        
        return losses["loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Dictionary containing:
                - x: State tensor
                - x_prime: Time derivative tensor
                - x2: Next state tensor
                - morphology: Morphology object (not used in forward pass)
                - task_type: Task type
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of validation metrics
        """
        # Extract data
        x = batch["x"]
        x_prime = batch["x_prime"]
        x2 = batch["x2"]
        
        # Compute losses
        losses = self._compute_losses(x, x_prime, x2)
        
        # Log metrics
        self.log("val_loss", losses["loss"], prog_bar=True)
        self.log("val_autoencoder_loss", losses["autoencoder_loss"])
        self.log("val_physics_loss", losses["physics_loss"])
        
        if batch_idx == 0:
            logger.info(f"Validation: val_loss={losses['loss']:.4f}, "
                       f"autoencoder_loss={losses['autoencoder_loss']:.4f}, "
                       f"physics_loss={losses['physics_loss']:.4f}")
        
        return losses
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Dictionary containing:
                - x: State tensor
                - x_prime: Time derivative tensor
                - x2: Next state tensor
                - morphology: Morphology object (not used in forward pass)
                - task_type: Task type
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of test metrics
        """
        # Extract data
        x = batch["x"]
        x_prime = batch["x_prime"]
        x2 = batch["x2"]
        
        # Compute losses
        losses = self._compute_losses(x, x_prime, x2)
        
        # Log metrics
        self.log("test_loss", losses["loss"])
        self.log("test_autoencoder_loss", losses["autoencoder_loss"])
        self.log("test_physics_loss", losses["physics_loss"])
        
        logger.info(f"Test: test_loss={losses['loss']:.4f}, "
                   f"autoencoder_loss={losses['autoencoder_loss']:.4f}, "
                   f"physics_loss={losses['physics_loss']:.4f}")
        
        return losses
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        logger.info(f"Configured Adam optimizer with lr={self.learning_rate}, weight_decay={self.weight_decay}")
        
        return optimizer

# Example usage
def train_model(
    input_dim: int = 12,
    hidden_dim: int = 256,
    latent_dim: int = 128,
    batch_size: int = 64,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    alpha: float = 0.7,
    beta: float = 0.3,
    simulation_steps: int = 10,
    bimanual_prob: float = 0.3,
    task_weights: Dict[str, float] = None
):
    """
    Train the morphology embedder.
    
    Args:
        input_dim: Dimension of the input state
        hidden_dim: Dimension of hidden layers
        latent_dim: Dimension of the latent space
        batch_size: Batch size
        max_epochs: Maximum number of epochs
        learning_rate: Learning rate
        alpha: Weight for autoencoder loss
        beta: Weight for physics loss
        simulation_steps: Number of simulation steps
        bimanual_prob: Probability of generating bimanual morphologies
        task_weights: Weights for different tasks
    """
    # Create data module
    data_module = MultiTaskDataModule(
        batch_size=batch_size,
        simulation_steps=simulation_steps,
        bimanual_prob=bimanual_prob,
        task_weights=task_weights or {"autoencoder": alpha, "physics": beta}
    )
    
    # Create model
    model = MorphologyEmbedder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        alpha=alpha,
        beta=beta
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=pl.loggers.TensorBoardLogger("logs/", name="morphology_embedder"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/",
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ]
    )
    
    # Train model
    logger.info("Starting training")
    trainer.fit(model, data_module)
    logger.info("Training completed")
    
    # Test model
    logger.info("Starting testing")
    trainer.test(model, data_module)
    logger.info("Testing completed")
    
    return model, trainer

if __name__ == "__main__":
    # Example usage
    train_model(
        input_dim=12,
        hidden_dim=256,
        latent_dim=128,
        batch_size=64,
        max_epochs=100,
        learning_rate=1e-4,
        alpha=0.7,
        beta=0.3,
        simulation_steps=10,
        bimanual_prob=0.3
    )
