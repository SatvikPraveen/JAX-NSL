Models API
==========

The ``models`` package contains Flax ``nn.Module`` implementations for MLP, CNN, and Transformer architectures.

.. contents:: Modules
   :local:
   :depth: 1

models.mlp
----------

Multi-layer perceptron.

.. automodule:: models.mlp
   :members:
   :undoc-members:
   :show-inheritance:

**Usage**:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from models.mlp import MLP, create_mlp

   key = jax.random.PRNGKey(0)
   mlp = MLP(features=[128, 64, 10], activation="relu")

   x = jnp.ones((4, 784))
   params = mlp.init(key, x)
   logits = mlp.apply(params, x)   # shape (4, 10)

models.cnn
----------

Convolutional neural network.

.. automodule:: models.cnn
   :members:
   :undoc-members:
   :show-inheritance:

**Usage**:

.. code-block:: python

   from models.cnn import CNN

   model = CNN(features=[32, 64], num_classes=10)
   x = jnp.ones((1, 28, 28, 1))   # NHWC format
   params = model.init(key, x)
   logits = model.apply(params, x)   # shape (1, 10)

models.transformer
------------------

Attention mechanisms and Transformer building blocks.

.. automodule:: models.transformer
   :members:
   :undoc-members:
   :show-inheritance:

**Key components**:

* ``scaled_dot_product_attention(Q, K, V)`` – core attention function.
* ``MultiHeadAttention`` – multi-head attention module.
* ``TransformerLayer`` – full encoder layer (attention + FFN + LayerNorm).
* ``Transformer`` – stacked encoder model.
