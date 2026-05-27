"""
============================================
Math utility functions (:mod:`rqutils.math`)
============================================

.. currentmodule:: rqutils.math

Math API
========

.. autosummary::
   :toctree: ../generated

   matrix_ufunc
   matrix_exp
   matrix_angle
"""
from collections.abc import Callable
import sys
import tempfile
from types import ModuleType
import numpy as np
from numpy.typing import ArrayLike, NDArray
import h5py


def matrix_ufunc(
    operator: Callable,
    mat: ArrayLike,
    hermitian: int | bool = 0,
    with_diagonals: bool = False,
    npmod: ModuleType = np,
    save_errors: bool = False
) -> NDArray | tuple[NDArray, NDArray]:
    """Apply a unitary-invariant unary matrix operator to an array of normal matrices.

    The argument `mat` must be an array of normal (i.e. square diagonalizable) matrices in the last
    two dimensions. This function unitary-diagonalizes the matrices, applies `operator` to the
    diagonals, and inverts the diagonalization.

    **Diagonalization and gradient**

    When using this function with an autodiff library (e.g. JAX), the gradient diverges when an
    input parameter controls off-diagonal elements of ``mat`` but ``mat`` is diagonal. Use an
    alternative function (that is hopefully available) in such cases:

    .. code-block:: python

        # Reshape the matrix to gather all off-diagonal elements to a block ([:, 1:])
        mat_dim = mat.shape[-1]
        diag_checker = mat.reshape(-1, mat_dim ** 2)
        # The very last element is a part of diagonal -> can ignore for this purpose
        diag_checker = diag_checker[:, :-1].reshape(-1, mat_dim - 1, mat_dim + 1)
        is_diagonal = ~jnp.any(diag_checker[:, :, 1:], axis=(1, 2))
        has_diagonal = jnp.any(is_diagonal)

        result = jax.lax.cond(has_diagonal,
                              alternative_X,
                              functools.partial(matrix_ufunc, X),
                              mat)

    Args:
        operator: Unary operator to be applied to the diagonals of ``mat``.
        mat: Array of normal matrices (shape (..., n, n)). No check on normality is performed.
        hermitian: 1 or True -> ``mat`` is Hermitian, -1 -> ``mat`` is anti-hermitian, 0 or
            False -> otherwise
        with_diagonals: If True, also return the array ``operator(eigenvalues)``.

    Returns:
        An array corresponding to `operator(mat)`. If `diagonals==True`, another array corresponding
        to `operator(eigvals)`.
    """
    try:
        if hermitian in (1, True):
            eigvals, eigcols = npmod.linalg.eigh(mat)
        elif hermitian == -1:
            eigvals, eigcols = npmod.linalg.eigh(1.j * mat)
            eigvals = -1.j * eigvals
        else:
            eigvals, eigcols = npmod.linalg.eig(mat)
    except:
        if save_errors:
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmpf:
                pass
            with h5py.File(tmpf.name, 'w') as out:
                out.create_dataset('matrices', data=mat)

            sys.stderr.write(f'Error in eigendecomposition. Matrix saved at {tmpf.name}\n')

        raise

    eigrows = npmod.conjugate(npmod.moveaxis(eigcols, -2, -1))
    op_eigvals = operator(eigvals)
    op_mat = npmod.matmul(eigcols * op_eigvals[..., None, :], eigrows)
    if with_diagonals:
        return op_mat, op_eigvals

    return op_mat


def matrix_exp(
    mat: ArrayLike,
    hermitian: int | bool = 0,
    with_diagonals: bool = False,
    npmod: ModuleType = np,
    save_errors: bool = False
) -> NDArray:
    """`matrix_ufunc(exp, ...)`"""
    return matrix_ufunc(npmod.exp, mat, hermitian=hermitian, with_diagonals=with_diagonals,
                        npmod=npmod, save_errors=save_errors)


def matrix_angle(
    mat: ArrayLike,
    hermitian: int | bool = 0,
    with_diagonals: bool = False,
    npmod: ModuleType = np,
    save_errors=False
) -> NDArray:
    """`matrix_ufunc(angle, ...)`"""
    return matrix_ufunc(npmod.angle, mat, hermitian=hermitian, with_diagonals=with_diagonals,
                        npmod=npmod, save_errors=save_errors)
