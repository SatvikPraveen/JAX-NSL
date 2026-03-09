Changelog
=========

All notable changes to JAX-NSL are documented here.
The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_.

Unreleased
----------

Added
~~~~~

* ``check_finite`` utility in ``core.arrays``.
* ``stable_logsumexp``, ``stable_softmax``, ``safe_sqrt``, ``numerical_gradient`` in ``core.numerics``.
* ``compute_gradient``, ``compute_jacobian``, ``compute_hessian`` convenience wrappers in ``autodiff.grad_jac_hess``.
* ``smooth_abs_vjp`` and ``smooth_abs_jvp`` custom differentiation examples.
* ``sync_gradients``, ``shard_array``, ``distributed_dot``, ``sync_batch_stats`` in ``parallel``.
* ``jit_with_static``, ``efficient_jit``, ``benchmark_jit`` in ``transforms.jit_utils``.
* ``batched_matmul``, ``batched_gradient``, ``parallel_apply`` in ``transforms.vmap_utils``.
* ``cumulative_sum``, ``solve_ode`` in ``transforms.scan_utils``.
* ``clip_gradient`` convenience wrapper in ``transforms.control_flow``.
* Full test suites: ``test_core``, ``test_models``, ``test_linalg``, ``test_training``.
* Shared ``conftest.py`` with pytest fixtures.
* ``pytest.ini`` with ``pythonpath = src``.
* ``pyproject.toml`` with black, isort, mypy, ruff configuration.
* ``Makefile`` with ``install``, ``test``, ``lint``, ``format``, ``typecheck``, ``docs``, ``clean`` targets.
* GitHub Actions CI workflow (lint, typecheck, test on Python 3.9/3.10/3.11).
* Comprehensive documentation: quickstart, tutorials, API reference, examples, contributing, changelog.

[0.1.0] – Initial Release
--------------------------

Added
~~~~~

* Core modules: ``core.arrays``, ``core.numerics``, ``core.prng``.
* Autodiff modules: ``autodiff.grad_jac_hess``, ``autodiff.custom_vjp``, ``autodiff.custom_jvp``.
* Transform modules: ``transforms.jit_utils``, ``transforms.vmap_utils``, ``transforms.scan_utils``, ``transforms.control_flow``.
* Linear algebra modules: ``linalg.ops``, ``linalg.solvers``.
* Model modules: ``models.mlp``, ``models.cnn``, ``models.transformer``.
* Training modules: ``training.losses``, ``training.optimizers``, ``training.train_loop``.
* Parallel modules: ``parallel.pmap_utils``, ``parallel.pjit_utils``, ``parallel.collectives``.
* Utility modules: ``utils.benchmarking``, ``utils.tree_utils``.
* 21 educational Jupyter notebooks across 7 topic areas.
* Synthetic data generation scripts.
* Docker and docker-compose configuration.
* Sphinx documentation scaffold.
