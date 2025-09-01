# File location: jax-nsl/src/__init__.py

"""
JAX-NSL: Neural Scientific Learning with JAX

A comprehensive educational framework for learning JAX from fundamentals
to advanced neural network implementations and parallel computing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports for easy access
from . import core
from . import autodiff
from . import transforms
from . import linalg
from . import models
from . import training
from . import parallel
from . import utils

# Version and metadata
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "core",
    "autodiff",
    "transforms",
    "linalg", 
    "models",
    "training",
    "parallel",
    "utils"
]