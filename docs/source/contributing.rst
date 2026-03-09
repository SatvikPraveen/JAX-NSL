Contributing
============

Contributions are welcome! This document explains how to set up a development environment, run the test suite, and submit pull requests.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/<your-username>/JAX-NSL.git
      cd JAX-NSL

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate   # macOS / Linux
      # or: venv\Scripts\activate  # Windows

3. Install all extras:

   .. code-block:: bash

      make install-dev
      # equivalent: pip install -e ".[all]"

Running Tests
-------------

.. code-block:: bash

   make test             # full test suite
   make test-fast        # skip slow / GPU tests
   make test-cov         # with HTML coverage report

Code Style
----------

The project enforces consistent formatting:

.. code-block:: bash

   make format           # black + isort (auto-fix)
   make lint             # ruff (report only)
   make format-check     # used in CI – exits non-zero on failure

Type Checking
-------------

.. code-block:: bash

   make typecheck        # mypy src/

Pull Request Guidelines
-----------------------

1. Create a feature branch from ``main``: ``git checkout -b feat/my-feature``.
2. Write or update tests for your change.
3. Ensure all checks pass: ``make format-check lint typecheck test``.
4. Open a pull request with a clear title and description.

Reporting Issues
----------------

Please use `GitHub Issues <https://github.com/SatvikPraveen/JAX-NSL/issues>`_ to report bugs or request features. Include:

* A minimal reproducible example.
* Your Python, JAX, and jaxlib versions (``pip show jax jaxlib``).
* OS and hardware (CPU / GPU / TPU).

Code of Conduct
---------------

Be respectful and constructive. This project follows the `Contributor Covenant <https://www.contributor-covenant.org/>`_.
