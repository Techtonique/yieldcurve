import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
)
from typing import Optional

def smith_wilson_kernel(t: np.ndarray, u: np.ndarray, alpha: float = 0.1, ufr: Optional[float] = 0.03) -> np.ndarray:
    """
    Generate Smith-Wilson kernel matrix.
    
    Parameters
    ----------
    t : array-like
        Time points to evaluate at (shape: n_samples_X)
    u : array-like
        Nodal points (shape: n_samples_Y)
    alpha : float, default=0.1
        Mean reversion parameter
    ufr : float or None, default=0.03
        Ultimate forward rate. If None, no UFR adjustment is applied.
        
    Returns
    -------
    np.ndarray
        Wilson functions matrix of shape (n_samples_X, n_samples_Y)
    """
    # Convert inputs to numpy arrays
    t = np.asarray(t)
    u = np.asarray(u)
    
    # Create meshgrid for vectorized operations
    t_mat, u_mat = np.meshgrid(t, u, indexing='ij')
    
    # Calculate min and max of t and u
    min_u = np.minimum(t_mat, u_mat)
    max_u = t_mat + u_mat - min_u
    
    # Calculate Wilson functions
    wilson_term = (
        alpha * min_u - 0.5 * np.exp(-alpha * max_u) * 
        (np.exp(alpha * min_u) - np.exp(-alpha * min_u))
    )
    
    if ufr is not None:
        # Apply UFR adjustment if ufr is provided
        return np.exp(-ufr * (t_mat + u_mat)) * wilson_term
    else:
        # Return without UFR adjustment
        return wilson_term

def generate_kernel(X, kernel_type='rbf', nodal_points=None, **kwargs):
    """Generate kernel matrix using sklearn.gaussian_process.kernels or custom kernels."""
    # Convert input to numpy array
    X = np.asarray(X)
    # Reshape 1D array to 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    # Get nodal points (training points)
    if nodal_points is None:
        nodal_points = X
    
    # Create kernel based on type
    if kernel_type == 'smithwilson':
        alpha = kwargs.get('alpha', 0.1)
        ufr = kwargs.get('ufr', 0.03)
        return smith_wilson_kernel(X.flatten(), nodal_points.flatten(), alpha, ufr)
    else:
        # Filter kwargs for sklearn kernels
        sklearn_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['length_scale', 'nu', 'alpha']}
        
        if kernel_type == 'rbf':
            kernel = RBF(**sklearn_kwargs)
        elif kernel_type == 'matern':
            kernel = Matern(**sklearn_kwargs)
        elif kernel_type == 'rationalquadratic':
            kernel = RationalQuadratic(**sklearn_kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
            
        # For sklearn kernels, compute K(X, nodal_points)
        return kernel(X, nodal_points)