#!/bin/bash

# JAX-NSL (Numerics & Systems Lab) Project Structure Generator
# This script creates the complete directory structure and files for the JAX-NSL project

echo "🚀 Creating JAX-NSL project structure..."

# Create root directory
mkdir -p jax-nsl
cd jax-nsl

# Create .github workflows
mkdir -p .github/workflows
touch .github/workflows/ci.yml
touch .github/workflows/docs.yml

# Create root files
touch .gitignore
touch PROJECT_STRUCTURE.md
touch README.md
touch requirements.txt
touch setup.py

# Create data directory
mkdir -p data/synthetic

# Create docker directory
mkdir -p docker
touch docker/docker-compose.yml
touch docker/Dockerfile

# Create docs directory
mkdir -p docs/source
touch docs/conf.py

# Create notebooks structure
echo "📓 Creating notebooks structure..."

# 01_fundamentals
mkdir -p notebooks/01_fundamentals
touch notebooks/01_fundamentals/01_arrays_and_prng.ipynb
touch notebooks/01_fundamentals/02_autodiff_basics.ipynb
touch notebooks/01_fundamentals/03_custom_vjp_jvp.ipynb
touch notebooks/01_fundamentals/04_control_flow_scan.ipynb

# 02_linear_algebra
mkdir -p notebooks/02_linear_algebra
touch notebooks/02_linear_algebra/05_matrix_ops.ipynb
touch notebooks/02_linear_algebra/06_iterative_solvers.ipynb
touch notebooks/02_linear_algebra/07_numerical_stability.ipynb

# 03_neural_networks
mkdir -p notebooks/03_neural_networks
touch notebooks/03_neural_networks/08_mlp_from_scratch.ipynb
touch notebooks/03_neural_networks/09_cnn_minimal.ipynb
touch notebooks/03_neural_networks/10_attention_from_scratch.ipynb

# 04_training_optimization
mkdir -p notebooks/04_training_optimization
touch notebooks/04_training_optimization/11_optimizers_in_jax.ipynb
touch notebooks/04_training_optimization/12_loss_functions.ipynb
touch notebooks/04_training_optimization/13_training_loops.ipynb

# 05_parallelism
mkdir -p notebooks/05_parallelism
touch notebooks/05_parallelism/14_pmap_basics.ipynb
touch notebooks/05_parallelism/15_pjit_and_sharding.ipynb
touch notebooks/05_parallelism/16_collectives.ipynb

# 06_special_topics
mkdir -p notebooks/06_special_topics
touch notebooks/06_special_topics/17_differentiable_odes.ipynb
touch notebooks/06_special_topics/18_probabilistic_gradients.ipynb
touch notebooks/06_special_topics/19_research_tricks.ipynb

# capstone_projects
mkdir -p notebooks/capstone_projects
touch notebooks/capstone_projects/20_physics_informed_nn.ipynb
touch notebooks/capstone_projects/21_large_scale_training.ipynb

# Create src structure
echo "🐍 Creating src module structure..."

# Root src
mkdir -p src
touch src/__init__.py

# core module
mkdir -p src/core
touch src/core/__init__.py
touch src/core/arrays.py
touch src/core/prng.py
touch src/core/numerics.py

# autodiff module
mkdir -p src/autodiff
touch src/autodiff/__init__.py
touch src/autodiff/grad_jac_hess.py
touch src/autodiff/custom_vjp.py
touch src/autodiff/custom_jvp.py

# transforms module
mkdir -p src/transforms
touch src/transforms/__init__.py
touch src/transforms/jit_utils.py
touch src/transforms/vmap_utils.py
touch src/transforms/scan_utils.py
touch src/transforms/control_flow.py

# linalg module
mkdir -p src/linalg
touch src/linalg/__init__.py
touch src/linalg/ops.py
touch src/linalg/solvers.py

# models module
mkdir -p src/models
touch src/models/__init__.py
touch src/models/mlp.py
touch src/models/cnn.py
touch src/models/transformer.py

# training module
mkdir -p src/training
touch src/training/__init__.py
touch src/training/losses.py
touch src/training/optimizers.py
touch src/training/train_loop.py

# parallel module
mkdir -p src/parallel
touch src/parallel/__init__.py
touch src/parallel/pmap_utils.py
touch src/parallel/pjit_utils.py
touch src/parallel/collectives.py

# utils module
mkdir -p src/utils
touch src/utils/__init__.py
touch src/utils/benchmarking.py
touch src/utils/tree_utils.py

# Create tests structure
echo "🧪 Creating tests structure..."
mkdir -p tests
touch tests/test_autodiff.py
touch tests/test_transforms.py
touch tests/test_parallel.py
touch tests/test_numerics.py

echo "✅ JAX-NSL project structure created successfully!"
echo ""
echo "📊 Project Summary:"
echo "   📁 Total directories: $(find . -type d | wc -l)"
echo "   📄 Total files: $(find . -type f | wc -l)"
echo "   📓 Notebooks: $(find notebooks -name "*.ipynb" | wc -l)"
echo "   🐍 Python modules: $(find src -name "*.py" | wc -l)"
echo "   🧪 Test files: $(find tests -name "*.py" | wc -l)"
echo ""
echo "🎯 Next steps:"
echo "   1. cd jax-nsl"
echo "   2. Initialize git: git init"
echo "   3. Create virtual environment: python -m venv venv"
echo "   4. Activate environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "   5. Install JAX and dependencies"
echo ""
echo "🔥 Ready to start your JAX mastery journey!"