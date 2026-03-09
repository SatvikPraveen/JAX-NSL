Capstone Projects
=================

Notebooks 20–21 are end-to-end projects that integrate concepts from all earlier notebooks.

.. toctree::
   :maxdepth: 1

Notebook 20 – Physics-Informed Neural Networks
------------------------------------------------

**File**: ``notebooks/capstone_projects/20_physics_informed_nn.ipynb``

Topics covered:

* Problem formulation: solving a 1-D PDE (heat equation) with a neural network.
* Residual loss on the PDE interior and boundary-condition loss.
* Automatic differentiation to compute spatial derivatives.
* Training loop with adaptive weighting of loss terms.
* Visualising the predicted solution vs. the analytical solution.

Notebook 21 – Large-Scale Training
------------------------------------

**File**: ``notebooks/capstone_projects/21_large_scale_training.ipynb``

Topics covered:

* Building a scalable data pipeline with ``tf.data`` or ``grain``.
* Multi-device training with ``pjit`` + ``Mesh``.
* Checkpoint save and restore with Orbax.
* Mixed-precision training (``bfloat16`` activations, ``float32`` parameters).
* Tracking experiments with a structured ``TrainState`` and rich progress bars.
* Evaluating throughput: tokens/sec, samples/sec.
