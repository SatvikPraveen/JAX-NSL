.
├── .github
│   └── workflows
│       ├── ci.yml
│       └── docs.yml
├── .gitignore
├── create_jax_nsl_structure.sh
├── data
│   └── synthetic
│       └── generate_data.py
├── docker
│   ├── docker-compose.yml
│   └── Dockerfile
├── docs
│   ├── conf.py
│   └── source
│       ├── index.rst
│       └── installation.rst
├── notebooks
│   ├── 01_fundamentals
│   │   ├── 01_arrays_and_prng.ipynb
│   │   ├── 02_autodiff_basics.ipynb
│   │   ├── 03_custom_vjp_jvp.ipynb
│   │   └── 04_control_flow_scan.ipynb
│   ├── 02_linear_algebra
│   │   ├── 05_matrix_ops.ipynb
│   │   ├── 06_iterative_solvers.ipynb
│   │   └── 07_numerical_stability.ipynb
│   ├── 03_neural_networks
│   │   ├── 08_mlp_from_scratch.ipynb
│   │   ├── 09_cnn_minimal.ipynb
│   │   └── 10_attention_from_scratch.ipynb
│   ├── 04_training_optimization
│   │   ├── 11_optimizers_in_jax.ipynb
│   │   ├── 12_loss_functions.ipynb
│   │   └── 13_training_loops.ipynb
│   ├── 05_parallelism
│   │   ├── 14_pmap_basics.ipynb
│   │   ├── 15_pjit_and_sharding.ipynb
│   │   └── 16_collectives.ipynb
│   ├── 06_special_topics
│   │   ├── 17_differentiable_odes.ipynb
│   │   ├── 18_probabilistic_gradients.ipynb
│   │   └── 19_research_tricks.ipynb
│   └── capstone_projects
│       ├── 20_physics_informed_nn.ipynb
│       └── 21_large_scale_training.ipynb
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── autodiff
│   │   ├── __init__.py
│   │   ├── custom_jvp.py
│   │   ├── custom_vjp.py
│   │   └── grad_jac_hess.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── arrays.py
│   │   ├── numerics.py
│   │   └── prng.py
│   ├── linalg
│   │   ├── __init__.py
│   │   ├── ops.py
│   │   └── solvers.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── mlp.py
│   │   └── transformer.py
│   ├── parallel
│   │   ├── __init__.py
│   │   ├── collectives.py
│   │   ├── pjit_utils.py
│   │   └── pmap_utils.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── optimizers.py
│   │   └── train_loop.py
│   ├── transforms
│   │   ├── __init__.py
│   │   ├── control_flow.py
│   │   ├── jit_utils.py
│   │   ├── scan_utils.py
│   │   └── vmap_utils.py
│   └── utils
│       ├── __init__.py
│       ├── benchmarking.py
│       └── tree_utils.py
└── tests
    ├── test_autodiff.py
    ├── test_numerics.py
    ├── test_parallel.py
    └── test_transforms.py

26 directories, 71 files
