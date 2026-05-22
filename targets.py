"""
Target distribution for the MCMC demo.

Each Target object should expose:
    name        : str
    description : str
    U(x)        : potential energy, U(x) ~= -p(x)
    grad_U(x)   : gradient of U
    x0          : starting point
    bounds      : (xmin, xmax, ymin, ymax) for plotting
"""

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


@dataclass
class Target:
    name: str
    description: str
    U: Callable[[np.ndarray], float]
    grad_U: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    bounds: Tuple[float, float, float, float]


def _2d_normal(rho=0.60):
    """
    Bivariate Gaussian with correlation rho, where
    U(x) = -log p(x) + const = 0.5 * x^T Sigma^-1 x
    for one x.
    """
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)

    def U(x):
        return 0.5 * x @ Sigma_inv @ x

    def grad_U(x):
        return Sigma_inv @ x

    return Target(
        name='Bivariate Gaussian',
        description=f'Bivariate Gaussian with correlation rho={rho}.',
        U=U, grad_U=grad_U,
        x0=np.array([-2.5, -2.5]),
        bounds=(-5.0, 5.0, -5.0, 5.0),
    )

TARGETS = {'2d_normal': _2d_normal()}
