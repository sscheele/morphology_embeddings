This is a research project aimed at producing morphology embeddings for robotic manipulators.

# Tech Stack
- Mechanics: done with the `mujoco` library
- Deep Learning: pytorch lightning
- Experiment tracking: mlflow

The environment is a prefixed conda environment located at {project_root}/robot_env.

# Style
Log important milestones relevant to a user using the `logging` library. Import the `logger` variable from the `setup_logger.py` file and use that as your logger.

We will make liberal use of the `lightning` library to make sure runs and data are reproducible.

# Approach
The most important aspects of the approach are the data, architecture, and training

## Data
We will generate data on-the-fly by randomly generating morphologies and running their dynamics through MuJoCo.

Random morphology generation: we will assume only revolute joints and rigid links. Furthermore, joints may only be at the terminus of a link. A morphology may be bimanual, but bimanual morphologies will always be symmetrical (identical arms separated by not more than the length of the longest link). We assume joint-space velocity control.

We gather data on the dynamics of the geometry by initializing it to a random start state with random joint velocities and playing it forward.

Morphology libraries are located in `morph_embed.morphology` and the most notable classes and function can be imported elsewhere as `morph_embed.Morphology`, `morph_embed.BimanualMorphology`, `morph_embed.mujoco_fk`

## Architecture
The embedder will implement a constrained attention module in which each joint may only attend to its ancestors in the kinematic tree. This will also help the network to distinguish bimanual from single-arm manipulators. We adopt an autoencoder-type architecture with a physics network in the middle. 

## Training
We'll aim for curriculum learning of some sort with a number of ancillary losses. To start out, we'll train autoencoder+forward kinematics. The full loss function will be a linear combination of the autoencoder and FK losses. We want the autoencoder to learn to encode _both_ morphology and state because both are relevant to tasks we might want to accomplish in the compressed state space.

If we have an oracle function `fk` which we're trying to approximate with network `F`, and the autoencoder and decoder are `A` and `A'`, then the loss would be:

$$\alpha ||A'(A(M, x, x')) - [M, x, x']||^2 - \beta ||F(A(M, x, x')) - fk(x, M)||^2$$

Where $x$ is the state and $M$ is the morphology.