Training and Optimisation Notebooks
=====================================

Notebooks 11–13 cover the full training lifecycle with Optax and Flax.

.. toctree::
   :maxdepth: 1

Notebook 11 – Optimisers in JAX
---------------------------------

**File**: ``notebooks/04_training_optimization/11_optimizers_in_jax.ipynb``

Topics covered:

* SGD, momentum, Nesterov momentum.
* Adam and AdamW (weight decay).
* Gradient clipping by value and by global norm.
* Learning-rate schedules: cosine annealing, linear warmup, step decay.
* Optax ``chain`` for composing transforms.

Notebook 12 – Loss Functions
------------------------------

**File**: ``notebooks/04_training_optimization/12_loss_functions.ipynb``

Topics covered:

* Softmax cross-entropy and label smoothing.
* Binary cross-entropy.
* Mean squared error and Huber / smooth-L1 loss.
* Focal loss for class-imbalanced datasets.
* KL divergence and contrastive losses.

Notebook 13 – Training Loops
------------------------------

**File**: ``notebooks/04_training_optimization/13_training_loops.ipynb``

Topics covered:

* ``TrainState`` dataclass (params, optimizer state, step counter).
* JIT-compiled ``train_step`` and ``eval_step``.
* Validation loop with accuracy and loss tracking.
* Early stopping with patience.
* Model checkpointing with Orbax / ``flax.training.checkpoints``.
* Learning-rate scheduling inside the training loop.
* Logging metrics with tqdm and rich.
