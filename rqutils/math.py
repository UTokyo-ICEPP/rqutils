"""
===================================================
Useful quantum math functions (:mod:`rqutils.math`)
===================================================

.. currentmodule:: rqutils.math

Math API
========

.. autosummary::
   :toctree: ../generated

   matrix_ufunc
   matrix_exp
   matrix_angle
"""

from typing import Callable, Tuple, Any, Union
import sys
import tempfile
import pickle
import numpy as np
try:
    import h5py
except ImportError:
    has_h5py = False
else:
    has_h5py = True

from ._types import ndarray, array_like

def matrix_ufunc(
    op: Callable,
    mat: array_like,
    hermitian: Union[int, bool] = 0,
    with_diagonals: bool = False,
    npmod=np,
    save_errors=False
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    """Apply a unitary-invariant unary matrix operator to an array of normal matrices.

    The argument `mat` must be an array of normal (i.e. square diagonalizable) matrices in the last
    two dimensions. This function unitary-diagonalizes the matrices, applies `op` to the diagonals,
    and inverts the diagonalization.

    Args:
        op: Unary operator to be applied to the diagonals of `mat`.
        mat: Array of normal matrices (shape (..., n, n)). No check on normality is performed.
        hermitian: 1 or True -> `mat` is Hermitian, -1 -> `mat` is anti-hermitian, 0 or False -> otherwise
        with_diagonals: If True, also return the array `op(eigenvalues)`.

    Returns:
        An array corresponding to `op(mat)`. If `diagonals==True`, another array corresponding to `op(eigvals)`.
    """
    try:
        if hermitian == 1 or hermitian == True:
            eigvals, eigcols = npmod.linalg.eigh(mat)
        elif hermitian == -1:
            eigvals, eigcols = npmod.linalg.eigh(1.j * mat)
            eigvals = -1.j * eigvals
        else:
            eigvals, eigcols = npmod.linalg.eig(mat)
    except:
        if save_errors:
            if has_h5py:
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmpf:
                    pass

                with h5py.File(tmpf.name, 'w') as out:
                    out.create_dataset('matrices', data=mat)
            else:
                with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                    pickle.dump(mat, tmpf)

            sys.stderr.write(f'Error in eigendecomposition. Matrix saved at {tmpf.name}\n')

        raise

    eigrows = npmod.conjugate(npmod.moveaxis(eigcols, -2, -1))

    op_eigvals = op(eigvals)

    op_mat = npmod.matmul(eigcols * op_eigvals[..., None, :], eigrows)

    if with_diagonals:
        return op_mat, op_eigvals
    else:
        return op_mat

def matrix_exp(
    mat: array_like,
    hermitian: Union[int, bool] = 0,
    with_diagonals: bool = False,
    npmod=np,
    save_errors=False
) -> ndarray:
    """`matrix_ufunc(exp, ...)`"""
    return matrix_ufunc(npmod.exp, mat, hermitian=hermitian, with_diagonals=with_diagonals,
                        npmod=npmod, save_errors=save_errors)

def matrix_angle(
    mat: array_like,
    hermitian: Union[int, bool] = 0,
    with_diagonals: bool = False,
    npmod=np,
    save_errors=False
) -> ndarray:
    """`matrix_ufunc(angle, ...)`"""
    return matrix_ufunc(npmod.angle, mat, hermitian=hermitian, with_diagonals=with_diagonals,
                        npmod=npmod, save_errors=save_errors)
