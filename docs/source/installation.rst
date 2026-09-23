Installation
============

Requirements
------------

* Python 3.11 or newer
* JAX 0.8.1 or newer (the test-suite runs against current JAX; deprecated
  APIs are treated as errors so drift is caught early)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/SatvikPraveen/JAX-NSL.git
   cd JAX-NSL
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"          # package + test/lint tools
   pip install -e ".[notebook]"     # add jupyter, matplotlib, optax

``uv`` works too: ``uv venv .venv && uv pip install -e ".[dev,notebook]"``.

Accelerators
------------

JAX wheels are backend-specific; follow the
`official installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for your CUDA/ROCm/TPU version, then ``pip install -e .`` on top.

Several virtual devices on a laptop
-----------------------------------

The parallelism utilities and notebooks need more than one device. On CPU
you can ask XLA to expose several *virtual* devices - the test-suite does
this in ``tests/conftest.py``::

   import os
   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
   import jax  # must come after setting the flag
   jax.devices()  # 8 CpuDevices

Verify
------

.. code-block:: bash

   pytest -q                 # 275 tests, ~45 s on a laptop
   make notebooks            # execute all 21 notebooks (takes a few minutes)
