# morphology_embeddings
Aims to produce embeddings for various manipulator morphologies, and to induce a dynamics on the embedding space such that robotic control can be done for arbitrary morphologies in this space.

Also an experiment in LLM-driven development.

WIP

# TODO
- [ ] LLM wrote some shady code for handling transforms from the phantom root joint, review this
- [ ] Move positional encoding implementation out of the vectorization class

PRD reproduced below:

This is a research project aimed at producing morphology embeddings for robotic manipulators.

# Tech Stack
- Mechanics: done with the `mujoco` library
- Deep Learning: pytorch lightning
- Experiment tracking: mlflow

The environment is a prefixed conda environment located at {project_root}/robot_env.

# Style
Log liberally using the `logging` library. Import the `logger` variable from the `setup_logger.py` file and use that as your logger.

We will make liberal use of the `lightning` library to make sure runs and data are reproducible.

# Approach
The most important aspects of the approach are the data, architecture, and training

## Data
We will generate data on-the-fly by randomly generating morphologies and running their dynamics through MuJoCo.

Random morphology generation: we will assume only revolute joints and rigid links. Furthermore, joints may only be at the terminus of a link. A morphology may be bimanual, but bimanual morphologies will always be symmetrical (identical arms separated by not more than the length of the longest link).

We gather data on the dynamics of the geometry by initializing it to a random start state with random joint velocities and playing it forward.

We'll experiment with a morphology vectorization where each link in the morphology corresponds to a number of dimensions in the vectorization:

1. Offset of link base in parent's base frame (3-dimensional): in this project's morphologies, the base of one link starts at the tip of its parent link, so the offset should turn out to be zero in the x and y directions in practice, but if we end up doing more complex morphologies it could be important that the extra dimensions are there.
2. Axis of rotation (3-dimensional)
3. Positional encoding of parent joint (k-dimensional, although in practice k will usually take the value of 2)
4. End-effector flag (1D)
5. Joint type (one-hot, 2D for now, fixed or rotational)

We will handle bimanual robots by adding a root point halfway between the two arms and fixed joints connecting our root point to the base of each arm. The bimanual bots can then be treated as a single morphology for purposes of vectorization.

## Architecture
The embedder will implement a constrained attention module in which each joint may only attend to its ancestors in the kinematic tree. We adopt an autoencoder-type architecture with a physics network in the middle. 

## Training
Specifically, if we have an encoder network $A$, decoder $A'$, and physics network $F$, every training step will optimize a two-part loss function:

$$\alpha ||[x, x'] - A'(A(x, x'))||^2 + \beta ||x_2 - A'(F(A(x, x')))||^2$$

Where x' is the time derivative of x and x_2 is the result of the MuJoCo dynamics