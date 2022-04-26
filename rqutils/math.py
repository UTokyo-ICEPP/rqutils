from typing import Callable, Any, Union
import sys
import tempfile
import numpy as np
import h5py

def matrix_ufunc(
    op: Callable,
    mat: 'array_like',
    hermitian: Union[int, bool] = 0,
    with_diagonals: bool = False,
    npmod=np,
    save_errors=False
) -> 'ndarray':
    """Apply a unitary-invariant unary matrix operator to an array of normal matrices.
    
    The argument `mat` must be an array of normal matrices (in the last two dimensions). This function
    unitary-diagonalizes the matrices, applies `op` to the diagonals, and inverts the diagonalization.
    
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
            eigvals *= -1.j
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

    op_eigvals = op(eigvals)
    
    op_mat = npmod.matmul(eigcols * op_eigvals[..., None, :], eigrows)

    if with_diagonals:
        return op_mat, op_eigvals
    else:
        return op_mat

def matrix_exp(
    mat: Any,
    hermitian: Union[int, bool] = 0,
    with_diagonals: bool = False,
    npmod=np,
    save_errors=False
) -> 'array':
    return matrix_ufunc(npmod.exp, mat, hermitian=hermitian, with_diagonals=with_diagonals,
                        npmod=npmod, save_errors=save_errors)

def matrix_angle(
    mat: Any,
    hermitian: Union[int, bool] = 0,
    with_diagonals: bool = False,
    npmod=np,
    save_errors=False
) -> 'array':
    return matrix_ufunc(npmod.angle, mat, hermitian=hermitian, with_diagonals=with_diagonals,
                        npmod=npmod, save_errors=save_errors)

