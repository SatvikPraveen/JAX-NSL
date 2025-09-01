# File location: data/synthetic/generate_data.py

"""Generate small synthetic datasets for testing and examples."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from pathlib import Path

def generate_regression_data(key, n_samples=1000, n_features=10, noise_level=0.1):
    """Generate synthetic regression data."""
    key, subkey = jr.split(key)
    
    # Generate features
    X = jr.normal(subkey, (n_samples, n_features))
    
    # Generate true coefficients
    key, subkey = jr.split(key)
    true_coef = jr.normal(subkey, (n_features,))
    
    # Generate targets with noise
    key, subkey = jr.split(key)
    noise = jr.normal(subkey, (n_samples,)) * noise_level
    y = X @ true_coef + noise
    
    return X, y, true_coef

def generate_classification_data(key, n_samples=1000, n_features=10, n_classes=3):
    """Generate synthetic classification data."""
    key, subkey = jr.split(key)
    
    # Generate features
    X = jr.normal(subkey, (n_samples, n_features))
    
    # Generate class centers
    key, subkey = jr.split(key)
    centers = jr.normal(subkey, (n_classes, n_features)) * 2
    
    # Assign samples to closest centers
    distances = jnp.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    y = jnp.argmin(distances, axis=1)
    
    return X, y, centers

def generate_time_series(key, n_timesteps=1000, n_features=5):
    """Generate synthetic time series data."""
    key, subkey = jr.split(key)
    
    # Generate AR coefficients
    ar_coef = jr.uniform(subkey, (n_features, n_features)) * 0.8
    
    # Initialize series
    series = jnp.zeros((n_timesteps, n_features))
    
    key, subkey = jr.split(key)
    series = series.at[0].set(jr.normal(subkey, (n_features,)))
    
    # Generate autoregressive series
    for t in range(1, n_timesteps):
        key, subkey = jr.split(key)
        noise = jr.normal(subkey, (n_features,)) * 0.1
        series = series.at[t].set(ar_coef @ series[t-1] + noise)
    
    return series, ar_coef

def save_datasets():
    """Generate and save all synthetic datasets."""
    key = jr.PRNGKey(42)
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    # Regression data
    key, subkey = jr.split(key)
    X_reg, y_reg, coef_reg = generate_regression_data(subkey)
    np.savez(data_dir / 'regression_data.npz', 
             X=X_reg, y=y_reg, true_coef=coef_reg)
    
    # Save as CSV too
    reg_df = pd.DataFrame(np.array(X_reg), 
                         columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
    reg_df['target'] = np.array(y_reg)
    reg_df.to_csv(data_dir / 'regression_data.csv', index=False)
    
    # Classification data
    key, subkey = jr.split(key)
    X_cls, y_cls, centers_cls = generate_classification_data(subkey)
    np.savez(data_dir / 'classification_data.npz', 
             X=X_cls, y=y_cls, centers=centers_cls)
    
    # Time series data
    key, subkey = jr.split(key)
    ts_data, ar_coef = generate_time_series(subkey)
    np.savez(data_dir / 'timeseries_data.npz', 
             series=ts_data, ar_coef=ar_coef)
    
    # Save time series as CSV
    ts_df = pd.DataFrame(np.array(ts_data), 
                        columns=[f'series_{i}' for i in range(ts_data.shape[1])])
    ts_df.to_csv(data_dir / 'timeseries_data.csv', index=False)
    
    print("Synthetic datasets generated successfully!")

if __name__ == "__main__":
    save_datasets()