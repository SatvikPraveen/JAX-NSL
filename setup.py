# File location: jax-nsl/setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jax-nsl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Scientific Learning with JAX - Educational framework for learning JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/jax-nsl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "gpu": [
            "jaxlib[cuda12_pip]>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jax-nsl=jax_nsl.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "jax_nsl": ["data/synthetic/*"],
    },
    zip_safe=False,
)