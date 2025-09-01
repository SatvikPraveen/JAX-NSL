# JAX-NSL: Neural Scientific Learning with JAX

A comprehensive educational repository for learning JAX from fundamentals to advanced neural network implementations and parallel computing.

## Overview

JAX-NSL provides a structured learning path through JAX's core concepts, from basic array operations to large-scale distributed training. Each module includes theoretical foundations, practical implementations, and hands-on notebooks.

## Project Structure

```
jax-nsl/
├── src/                    # Core library modules
├── notebooks/              # Interactive learning materials
├── tests/                  # Unit tests
├── data/                   # Synthetic datasets
├── docs/                   # Documentation
└── docker/                 # Containerization
```

## Learning Path

### 1. Fundamentals (notebooks/01_fundamentals/)

- Arrays and PRNG systems
- Automatic differentiation basics
- Custom VJP/JVP operations
- Control flow with scan

### 2. Linear Algebra (notebooks/02_linear_algebra/)

- Matrix operations and decompositions
- Iterative solvers (CG, LBFGS)
- Numerical stability techniques

### 3. Neural Networks (notebooks/03_neural_networks/)

- MLP from scratch
- CNN implementations
- Attention mechanisms

### 4. Training & Optimization (notebooks/04_training_optimization/)

- Optimizers (SGD, Adam, AdamW)
- Loss functions and stability
- Training loop patterns

### 5. Parallelism (notebooks/05_parallelism/)

- Data parallelism with pmap
- Model parallelism with pjit
- Collective operations

### 6. Special Topics (notebooks/06_special_topics/)

- Differentiable ODEs
- Probabilistic gradients
- Research-level techniques

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Start with fundamentals
jupyter notebook notebooks/01_fundamentals/01_arrays_and_prng.ipynb
```

## Installation

### Local Setup

```bash
git clone https://github.com/SatvikPraveen/jax-nsl.git
cd jax-nsl
pip install -r requirements.txt
pip install -e .
```

### Docker Setup

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## Key Features

- **Pure JAX implementations** - No high-level frameworks, understand the fundamentals
- **Numerical stability focus** - Production-ready implementations
- **Parallel computing ready** - Multi-device and distributed examples
- **Research-oriented** - Advanced topics and cutting-edge techniques
- **Educational first** - Clear explanations and progressive complexity

## Requirements

- Python 3.8+
- JAX 0.4.0+
- NumPy, SciPy
- Jupyter for notebooks
- Optional: CUDA for GPU acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@misc{jax-nsl,
  title={JAX-NSL: Neural Scientific Learning with JAX},
  author={Your Name},
  year={2025},
  url={https://github.com/SatvikPraveen/jax-nsl}
}
```
