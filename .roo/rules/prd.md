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
Specifically, if we have an encoder network $A$, decoder $A'$, and physics network $F$, every training step will optimize a two-part loss function:

$$\alpha ||[x, x'] - A'(A(x, x'))||^2 + \beta ||x_2 - A'(F(A(x, x')))||^2$$

Where x' is the time derivative of x and x_2 is the result of the MuJoCo dynamics