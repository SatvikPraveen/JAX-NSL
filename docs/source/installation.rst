# File location: docs/source/installation.rst

Installation
============

This guide covers different ways to install JAX-NSL and its dependencies.

Requirements
------------

* Python 3.8+
* JAX 0.4.0+
* NumPy
* SciPy
* Matplotlib (for visualization)
* Jupyter (for notebooks)

Basic Installation
------------------

Install from PyPI (when available):

.. code-block:: bash

   pip install jax-nsl

Development Installation
------------------------

For development or to use the latest features:

.. code-block:: bash

   git clone https://github.com/your-repo/jax-nsl.git
   cd jax-nsl
   pip install -e .

This will install the package in "editable" mode, so changes to the source code are immediately available.

GPU Support
-----------

For CUDA support, install JAX with CUDA:

.. code-block:: bash

   # For CUDA 11.x
   pip install "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   
   # For CUDA 12.x
   pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

For TPU support:

.. code-block:: bash

   pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

Docker Installation
-------------------

Use the provided Docker setup:

.. code-block:: bash

   cd jax-nsl
   docker-compose up jax-nsl

This will build the image and start a Jupyter Lab server accessible at http://localhost:8888.

Conda Installation
------------------

Create a conda environment:

.. code-block:: bash

   conda create -n jax-nsl python=3.10
   conda activate jax-nsl
   pip install jax jaxlib
   pip install -e .

Virtual Environment
-------------------

Using venv:

.. code-block:: bash

   python -m venv jax-nsl-env
   source jax-nsl-env/bin/activate  # On Windows: jax-nsl-env\Scripts\activate
   pip install -r requirements.txt
   pip install -e .

Verification
------------

Test your installation:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_nsl.core import arrays
   
   # Check JAX backend
   print(f"JAX backend: {jax.default_backend()}")
   print(f"Available devices: {jax.devices()}")
   
   # Test basic functionality
   x = jnp.array([1., 2., 3.])
   y = x ** 2
   print(f"Test array: {y}")

If you see output without errors, your installation is successful!

Common Issues
-------------

**ImportError: No module named 'jax'**
   Install JAX: ``pip install jax jaxlib``

**CUDA not found**
   Install CUDA-enabled JAX following the GPU Support section above.

**Permission denied errors**
   Use ``--user`` flag: ``pip install --user jax-nsl``

**Version conflicts**
   Create a fresh virtual environment and install dependencies step by step.