"""GP functions.

It is expected that all function inputs will be numeric.
"""
import numpy as np

def add(X1, X2):
    """Return result of addition."""
    res = X1 + X2
    res[~np.isfinite(res)] = np.inf
    return res

def aq(X1, X2):
    """Return result of analytical quotient.
    
    The analytical quotient is as defined by Ni et al. in their paper 
    'The use of an analytic quotient operator in genetic programming':  
    `aq(x1, x2) = (x1) / (sqrt(1 + x2 ** (2)))`.
    """
    res = X1 / np.sqrt(1 + X2 ** 2)
    res[~np.isfinite(res)] = np.inf
    return res

def exp(X): 
    """Return result of exponentiation, base `e`."""
    return np.exp(X)

def log(X):
    """Return result of protected logarithm, base `e`.
    
    The argument of the logarithm is made positive.
    """
    res = np.log(np.abs(X))
    res[res == -np.inf] = 0
    return res

def mul(X1, X2):
    """Return result of multiplication."""
    res = X1 * X2
    res[~np.isfinite(res)] = np.inf
    return res

def sin(X):
    """Return result of sine."""
    res = np.sin(X)
    res[~np.isfinite(res)] = np.inf
    return res

def sqrt(X):
    """Return result of protected square root."""
    return np.sqrt(np.abs(X))

def sub(X1, X2):
    """Return result of subtraction."""
    res = X1 - X2
    res[~np.isfinite(res)] = np.inf
    return res

def tanh(X):
    """Return result of hyperbolic tangent."""
    return np.tanh(X)