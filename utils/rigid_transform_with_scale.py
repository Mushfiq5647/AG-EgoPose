#!/usr/bin/env python3
"""
Umeyama algorithm for rigid body transformation with scale
Based on: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py
"""

import numpy as np


def umeyama(src, dst, estimate_scale=True):
    """
    Estimate N-D similarity transformation with or without scaling.
    
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
        
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values if the problem is not well-conditioned.
        
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    
    num = src.shape[0]
    dim = src.shape[1]
    
    # Compute mean of source and destination
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    
    # Remove mean
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    # Eq. (38)-(39)
    A = dst_demean.T @ src_demean / num
    
    # Eq. (41)
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    
    T = np.eye(dim + 1, dtype=np.double)
    
    U, S, V = np.linalg.svd(A)
    
    # Eq. (42)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    
    if estimate_scale:
        # Eq. (43)
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0
    
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    
    return T


def umeyama_parameters(src, dst, estimate_scale=True):
    """
    Extract transformation parameters from Umeyama algorithm.
    
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
        
    Returns
    -------
    c : float
        Scale factor
    R : (N, N) array
        Rotation matrix
    t : (N,) array
        Translation vector
    """
    T = umeyama(src, dst, estimate_scale)
    
    if np.any(np.isnan(T)):
        return np.nan, np.nan, np.nan
    
    dim = T.shape[0] - 1
    
    # Extract scale from transformation matrix
    if estimate_scale:
        # Scale is the determinant of the rotation part
        R = T[:dim, :dim]
        c = np.linalg.det(R) ** (1.0 / dim)
        R = R / c
    else:
        c = 1.0
        R = T[:dim, :dim]
    
    # Extract translation
    t = T[:dim, dim]
    
    return c, R, t
