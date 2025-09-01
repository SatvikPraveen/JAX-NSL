# File location: docs/source/index.rst

JAX Neural Scientific Learning (JAX-NSL)
=========================================

Welcome to JAX-NSL, a comprehensive educational library for learning neural networks and scientific computing with JAX. This library provides hands-on notebooks, clean implementations, and practical examples covering everything from basic array operations to advanced parallel training strategies.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Notebooks:

   notebooks/01_fundamentals
   notebooks/02_linear_algebra
   notebooks/03_neural_networks
   notebooks/04_training_optimization
   notebooks/05_parallelism
   notebooks/06_special_topics
   notebooks/capstone_projects

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/core
   api/autodiff
   api/transforms
   api/linalg
   api/models
   api/training
   api/parallel
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/basic_usage
   examples/advanced_patterns
   examples/performance_tips

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog

Features
--------

* **Comprehensive Coverage**: From basic JAX operations to advanced neural network architectures
* **Educational Focus**: Clean, well-commented code with detailed explanations
* **Hands-on Learning**: Interactive notebooks with practical exercises
* **Production-Ready**: Efficient implementations suitable for research and production
* **Parallel Computing**: Examples of data and model parallelism with JAX
* **Modern Practices**: Current best practices in JAX and neural network development

Quick Example
-------------

Here's a simple example of training a neural network with JAX-NSL:

.. code-block:: python

   import jax.numpy as jnp
   from jax_nsl.models import MLP
   from jax_nsl.training import train_step
   from jax_nsl.core import PRNGSequence

   # Initialize model and data
   key = jax.random.PRNGKey(42)
   model = MLP([64, 32, 10])
   params = model.init(key, jnp.ones((1, 784)))
   
   # Training loop
   for batch in dataloader:
       params, loss = train_step(params, batch, model)
       print(f"Loss: {loss:.4f}")

Installation
------------

Install JAX-NSL with pip:

.. code-block:: bash

   pip install jax-nsl

Or for development:

.. code-block:: bash

   git clone https://github.com/your-repo/jax-nsl.git
   cd jax-nsl
   pip install -e .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`