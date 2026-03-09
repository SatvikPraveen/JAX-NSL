Training API
============

The ``training`` package provides loss functions, Optax-based optimisers, and complete training-loop utilities.

.. contents:: Modules
   :local:
   :depth: 1

training.losses
---------------

.. automodule:: training.losses
   :members:
   :undoc-members:
   :show-inheritance:

**Available losses**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``cross_entropy_loss(logits, labels)``
     - Softmax cross-entropy for classification.
   * - ``mse_loss(preds, targets)``
     - Mean squared error.
   * - ``huber_loss(preds, targets, delta)``
     - Huber (smooth L1) loss.
   * - ``focal_loss(logits, labels, gamma)``
     - Focal loss for class imbalance.
   * - ``binary_cross_entropy(logits, labels)``
     - BCE for binary classification.
   * - ``kl_divergence(p, q)``
     - KL divergence between two distributions.
   * - ``label_smoothing_cross_entropy``
     - Cross-entropy with label smoothing.

training.optimizers
-------------------

Thin wrappers around Optax optimisers with learning-rate schedules.

.. automodule:: training.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

training.train_loop
-------------------

``TrainState``, training-step helper, and evaluation utilities.

.. automodule:: training.train_loop
   :members:
   :undoc-members:
   :show-inheritance:

**Usage**:

.. code-block:: python

   from training.train_loop import create_train_state, train_step

   state = create_train_state(model, rng, learning_rate=1e-3, input_shape=(1, 784))

   for batch in dataloader:
       state, loss = train_step(state, batch)
