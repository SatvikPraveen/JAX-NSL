# Project structure

Tracked files (generated from `git ls-files`; `.venv/`, caches and build
outputs are ignored).

```
.
├── .github
│   └── workflows
│       └── ci.yml
├── data
│   └── synthetic
│       └── generate_data.py
├── docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs
│   ├── source
│   │   ├── api
│   │   │   ├── autodiff.rst
│   │   │   ├── core.rst
│   │   │   ├── linalg.rst
│   │   │   ├── models.rst
│   │   │   ├── parallel.rst
│   │   │   ├── training.rst
│   │   │   ├── transforms.rst
│   │   │   └── utils.rst
│   │   ├── examples
│   │   │   ├── advanced_patterns.rst
│   │   │   ├── basic_usage.rst
│   │   │   └── performance_tips.rst
│   │   ├── notebooks
│   │   │   ├── 01_fundamentals.rst
│   │   │   ├── 02_linear_algebra.rst
│   │   │   ├── 03_neural_networks.rst
│   │   │   ├── 04_training_optimization.rst
│   │   │   ├── 05_parallelism.rst
│   │   │   ├── 06_special_topics.rst
│   │   │   └── capstone_projects.rst
│   │   ├── changelog.rst
│   │   ├── contributing.rst
│   │   ├── index.rst
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   └── tutorials.rst
│   └── conf.py
├── notebooks
│   ├── 01_fundamentals
│   │   ├── 01_arrays_and_prng.ipynb
│   │   ├── 02_autodiff_basics.ipynb
│   │   ├── 03_custom_vjp_jvp.ipynb
│   │   └── 04_control_flow_scan.ipynb
│   ├── 02_linear_algebra
│   │   ├── 05_matrix_ops.ipynb
│   │   ├── 06_iterative_solvers.ipynb
│   │   └── 07_numerical_stability.ipynb
│   ├── 03_neural_networks
│   │   ├── 08_mlp_from_scratch.ipynb
│   │   ├── 09_cnn_minimal.ipynb
│   │   └── 10_attention_from_scratch.ipynb
│   ├── 04_training_optimization
│   │   ├── 11_optimizers_in_jax.ipynb
│   │   ├── 12_loss_functions.ipynb
│   │   └── 13_training_loops.ipynb
│   ├── 05_parallelism
│   │   ├── 14_pmap_basics.ipynb
│   │   ├── 15_pjit_and_sharding.ipynb
│   │   └── 16_collectives.ipynb
│   ├── 06_special_topics
│   │   ├── 17_differentiable_odes.ipynb
│   │   ├── 18_probabilistic_gradients.ipynb
│   │   └── 19_research_tricks.ipynb
│   └── capstone_projects
│       ├── 20_physics_informed_nn.ipynb
│       └── 21_large_scale_training.ipynb
├── src
│   └── jax_nsl
│       ├── autodiff
│       │   ├── __init__.py
│       │   ├── custom_jvp.py
│       │   ├── custom_vjp.py
│       │   ├── grad_jac_hess.py
│       │   └── implicit.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── arrays.py
│       │   ├── numerics.py
│       │   └── prng.py
│       ├── linalg
│       │   ├── __init__.py
│       │   ├── ops.py
│       │   └── solvers.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── cnn.py
│       │   ├── mlp.py
│       │   └── transformer.py
│       ├── parallel
│       │   ├── __init__.py
│       │   ├── collectives.py
│       │   ├── pjit_utils.py
│       │   └── pmap_utils.py
│       ├── training
│       │   ├── __init__.py
│       │   ├── losses.py
│       │   ├── optimizers.py
│       │   └── train_loop.py
│       ├── transforms
│       │   ├── __init__.py
│       │   ├── control_flow.py
│       │   ├── jit_utils.py
│       │   ├── scan_utils.py
│       │   └── vmap_utils.py
│       ├── utils
│       │   ├── __init__.py
│       │   ├── benchmarking.py
│       │   └── tree_utils.py
│       └── __init__.py
├── tests
│   ├── conftest.py
│   ├── test_autodiff.py
│   ├── test_core.py
│   ├── test_linalg.py
│   ├── test_models.py
│   ├── test_numerics.py
│   ├── test_parallel.py
│   ├── test_training.py
│   ├── test_transforms.py
│   └── test_utils.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── PROJECT_STRUCTURE.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

Key locations:

- `src/jax_nsl/` - the package (installed with `pip install -e .`).
- `tests/` - pytest suite; `conftest.py` creates 8 virtual CPU devices.
- `notebooks/` - 21 self-contained notebooks, executed in CI.
- `docs/` - Sphinx sources (`make docs`).
- `data/synthetic/generate_data.py` - small synthetic datasets for experiments.
- `docker/` - Jupyter Lab container.
