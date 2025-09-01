# File location: jax-nsl/src/training/losses.py

"""
Loss functions with numerical stability.

This module implements common loss functions used in deep learning
with emphasis on numerical stability and efficiency.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Union
from ..core.numerics import safe_log, logsumexp_stable, softmax_stable


def cross_entropy_loss(logits: jnp.ndarray,
                      labels: jnp.ndarray,
                      reduction: str = 'mean',
                      label_smoothing: float = 0.0) -> jnp.ndarray:
    """Cross-entropy loss with label smoothing.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        labels: True labels (batch_size,) for sparse or (batch_size, num_classes) for one-hot
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor
        
    Returns:
        Cross-entropy loss
    """
    if labels.ndim == 1:
        # Sparse labels - convert to one-hot
        num_classes = logits.shape[-1]
        labels_one_hot = jax.nn.one_hot(labels, num_classes)
    else:
        labels_one_hot = labels
    
    # Apply label smoothing
    if label_smoothing > 0:
        num_classes = labels_one_hot.shape[-1]
        smooth_labels = (1 - label_smoothing) * labels_one_hot + \
                       label_smoothing / num_classes
    else:
        smooth_labels = labels_one_hot
    
    # Compute log probabilities using logsumexp for stability
    log_probs = logits - logsumexp_stable(logits, axis=-1, keepdims=True)
    
    # Cross-entropy loss
    loss = -jnp.sum(smooth_labels * log_probs, axis=-1)
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def binary_cross_entropy(logits: jnp.ndarray,
                        labels: jnp.ndarray,
                        reduction: str = 'mean',
                        pos_weight: Optional[float] = None) -> jnp.ndarray:
    """Binary cross-entropy loss.
    
    Args:
        logits: Model predictions (any shape)
        labels: Binary labels (same shape as logits)
        reduction: 'mean', 'sum', or 'none'
        pos_weight: Weight for positive class
        
    Returns:
        Binary cross-entropy loss
    """
    # Use stable sigmoid cross-entropy
    max_val = jnp.maximum(-logits, 0)
    loss = logits - logits * labels + max_val + \
           jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))
    
    # Apply positive class weighting
    if pos_weight is not None:
        loss = loss * (pos_weight * labels + (1 - labels))
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def mse_loss(predictions: jnp.ndarray,
            targets: jnp.ndarray,
            reduction: str = 'mean') -> jnp.ndarray:
    """Mean squared error loss.
    
    Args:
        predictions: Model predictions
        targets: Target values
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        MSE loss
    """
    loss = (predictions - targets) ** 2
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def huber_loss(predictions: jnp.ndarray,
              targets: jnp.ndarray,
              delta: float = 1.0,
              reduction: str = 'mean') -> jnp.ndarray:
    """Huber loss (smooth L1 loss).
    
    Args:
        predictions: Model predictions
        targets: Target values
        delta: Threshold for switching between quadratic and linear loss
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Huber loss
    """
    residual = jnp.abs(predictions - targets)
    quadratic = 0.5 * residual ** 2
    linear = delta * (residual - 0.5 * delta)
    
    loss = jnp.where(residual <= delta, quadratic, linear)
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def focal_loss(logits: jnp.ndarray,
              labels: jnp.ndarray,
              alpha: float = 1.0,
              gamma: float = 2.0,
              reduction: str = 'mean') -> jnp.ndarray:
    """Focal loss for addressing class imbalance.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        labels: True labels (batch_size,)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Focal loss
    """
    # Convert to probabilities
    probs = softmax_stable(logits, axis=-1)
    
    # Get probabilities for true classes
    if labels.ndim == 1:
        # Sparse labels
        labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    else:
        labels_one_hot = labels
    
    p_t = jnp.sum(probs * labels_one_hot, axis=-1)
    
    # Focal loss computation
    alpha_t = alpha * labels_one_hot + (1 - alpha) * (1 - labels_one_hot)
    alpha_t = jnp.sum(alpha_t, axis=-1)
    
    focal_weight = alpha_t * (1 - p_t) ** gamma
    loss = -focal_weight * safe_log(p_t)
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def contrastive_loss(embeddings1: jnp.ndarray,
                    embeddings2: jnp.ndarray,
                    labels: jnp.ndarray,
                    margin: float = 1.0,
                    reduction: str = 'mean') -> jnp.ndarray:
    """Contrastive loss for similarity learning.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels: Binary labels (1 for similar, 0 for dissimilar)
        margin: Margin for dissimilar pairs
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Contrastive loss
    """
    # Euclidean distance
    distances = jnp.linalg.norm(embeddings1 - embeddings2, axis=-1)
    
    # Contrastive loss
    similar_loss = labels * distances ** 2
    dissimilar_loss = (1 - labels) * jnp.maximum(0, margin - distances) ** 2
    
    loss = 0.5 * (similar_loss + dissimilar_loss)
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def triplet_loss(anchor: jnp.ndarray,
                positive: jnp.ndarray,
                negative: jnp.ndarray,
                margin: float = 1.0,
                reduction: str = 'mean') -> jnp.ndarray:
    """Triplet loss for metric learning.
    
    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings
        negative: Negative embeddings
        margin: Margin between positive and negative distances
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Triplet loss
    """
    # Compute distances
    pos_dist = jnp.linalg.norm(anchor - positive, axis=-1)
    neg_dist = jnp.linalg.norm(anchor - negative, axis=-1)
    
    # Triplet loss
    loss = jnp.maximum(0, pos_dist - neg_dist + margin)
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def kl_divergence(p_logits: jnp.ndarray,
                 q_logits: jnp.ndarray,
                 reduction: str = 'mean') -> jnp.ndarray:
    """Kullback-Leibler divergence between two distributions.
    
    Args:
        p_logits: Logits for distribution p
        q_logits: Logits for distribution q
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        KL divergence D(p||q)
    """
    # Convert to log probabilities
    log_p = p_logits - logsumexp_stable(p_logits, axis=-1, keepdims=True)
    log_q = q_logits - logsumexp_stable(q_logits, axis=-1, keepdims=True)
    
    # KL divergence
    kl_div = jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)
    
    if reduction == 'mean':
        return jnp.mean(kl_div)
    elif reduction == 'sum':
        return jnp.sum(kl_div)
    else:
        return kl_div


def cosine_similarity_loss(embeddings1: jnp.ndarray,
                          embeddings2: jnp.ndarray,
                          labels: jnp.ndarray,
                          reduction: str = 'mean') -> jnp.ndarray:
    """Cosine similarity loss.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels: Target similarity scores (-1 to 1)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Cosine similarity loss
    """
    # Normalize embeddings
    norm1 = jnp.linalg.norm(embeddings1, axis=-1, keepdims=True)
    norm2 = jnp.linalg.norm(embeddings2, axis=-1, keepdims=True)
    
    embeddings1_norm = embeddings1 / (norm1 + 1e-8)
    embeddings2_norm = embeddings2 / (norm2 + 1e-8)
    
    # Cosine similarity
    cosine_sim = jnp.sum(embeddings1_norm * embeddings2_norm, axis=-1)
    
    # Mean squared error with target similarity
    loss = (cosine_sim - labels) ** 2
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def smooth_l1_loss(predictions: jnp.ndarray,
                  targets: jnp.ndarray,
                  beta: float = 1.0,
                  reduction: str = 'mean') -> jnp.ndarray:
    """Smooth L1 loss (used in object detection).
    
    Args:
        predictions: Model predictions
        targets: Target values
        beta: Threshold for switching between L1 and L2 loss
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Smooth L1 loss
    """
    diff = jnp.abs(predictions - targets)
    loss = jnp.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss


def dice_loss(predictions: jnp.ndarray,
             targets: jnp.ndarray,
             smooth: float = 1.0,
             reduction: str = 'mean') -> jnp.ndarray:
    """Dice loss for segmentation tasks.
    
    Args:
        predictions: Model predictions (probabilities)
        targets: Binary target masks
        smooth: Smoothing factor to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Dice loss
    """
    # Flatten spatial dimensions
    pred_flat = predictions.reshape(predictions.shape[0], -1)
    target_flat = targets.reshape(targets.shape[0], -1)
    
    # Compute Dice coefficient
    intersection = jnp.sum(pred_flat * target_flat, axis=1)
    union = jnp.sum(pred_flat, axis=1) + jnp.sum(target_flat, axis=1)
    
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice_coeff
    
    if reduction == 'mean':
        return jnp.mean(loss)
    elif reduction == 'sum':
        return jnp.sum(loss)
    else:
        return loss