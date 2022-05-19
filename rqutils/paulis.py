r"""
==================================================
Generalized Pauli matrices (:mod:`rqutils.paulis`)
==================================================

.. currentmodule:: rqutils.paulis

Fundamentals
============

Generalized :math:`n`-dimensional Pauli matrices :math:`\lambda^{(n)}_{k}` (:math:`0 \leq k \leq n^2 - 1`)
are defined recursively:

- :math:`\lambda^{(n)}_{0} = \sqrt{\frac{2}{n}} \mathrm{diag}(1, \dots, 1, 1)`
- :math:`\lambda^{(n)}_{k} = \mathrm{blkdiag}(\lambda^{(n-1)}_{k}, 0)` for :math:`1 \leq k < (n-1)^2`
- :math:`(\lambda^{(n)}_{(n-1)^2 + k})_{ab} = \xi_k \delta_{k//2, a}\delta_{n-1, b} + \eta_k \delta_{n-1, a}\delta_{k//2, b}`
  for :math:`0 \leq k < 2(n-1)`, with :math:`\xi_k = \eta_k = 1` (:math:`k` even) and :math:`-\xi_k = \eta_k = i` (:math:`k` odd)
- :math:`\lambda^{(n)}_{n^2-1} = \sqrt{\frac{2}{n(n-1)}} \mathrm{diag}(1, \dots, 1, -n+1)`

These matrices satisfy the normalization condition

.. math::

    \mathrm{tr}(\lambda^{(n)}_k \lambda^{(n)}_l) = 2 \delta_{k, l}

and thus form an orthonormal basis for the space of :math:`n`-dimensional Hermitian matrices.

Implications of the normalization
---------------------------------

Any :math:`n`-dimensional Hermitian matrix :math:`H` can be decomposed into a form

.. math::

    H = \sum_{k=0}^{n^2-1} \nu_k \lambda^{(n)}_k.

To extract the coefficient :math:`\nu_k`, one needs to compute

.. math::

    \nu_k = \frac{1}{2} \mathrm{tr}(\lambda^{(n)}_k H),

i.e., divide the product trace by 2.

Also, note that :math:`\lambda^{(n)}_{0}` is *not* the :math:`n`-dimensional identity matrix but differ
from it by a factor :math:`\sqrt{\frac{2}{n}}`.


Pauli products
==============

A physical composite system of :math:`s` subsystems is usually better described in terms of a tensor
product of :math:`s` Hamiltonians each of dimension :math:`n_i (i=1, \dots, s)`, rather than a single
Hamiltonian of :math:`N := \prod_{i=1}^{s} n^i` dimensions. A natural decomposition of the former
would be in terms of tensor products of :math:`s` Pauli matrices

.. math::

    \Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s} = \frac{1}{2^{s-1}} \bigotimes_{i=1}^{s} \lambda^{(n_i)}_{k_i},

which constitute an orthonormal basis of the space of :math:`N`-dimensional Hermitian matrices with
a rather awkward normalization

.. math::

    \mathrm{tr}(\Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s} \Lambda^{(n_1 \dots n_s)}_{l_1 \dots l_s}) = 2 \frac{1}{2^{s-1}} \prod_i \delta_{k_i, l_i}.

The full `s`-body Hamiltonian :math:`H` is decomposed into

.. math::

    H = \sum_{k_1 \dots k_s} \nu_{k_1 \dots k_s} \Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s},

and the component :math:`\nu_{k_1 \dots k_s}` is extracted by

.. math::

    \nu_{k_1 \dots k_s} = 2^{s-2} \mathrm{tr}(\Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s} H).


Dimension truncation
====================

Thanks to the recursive definition of the Pauli matrices, the decomposition of an :math:`m`-dimensional
submatrix of an :math:`n`-dimensional Hermitian matrix is mostly trivially obtained from the components
of the latter:

If

.. math::

    H^{(n)} = \sum_{k=0}^{n^2-1} \nu_k \lambda^{(n)}_k

then the truncated matrix :math:`\bar{H}^{(m)}` is

.. math::

    \bar{H}^{(m)} = \sum_{k=0}^{m^2-1} \bar{\nu}_k \lambda^{(m)}_k,

with :math:`\bar{\nu}_k = \nu_k` for :math:`1 \leq k \leq m^2 - 1`. For :math:`k=0`, however, we need to
consider the projection of the diagonal matrices:

.. math::

    \mathrm{tr}_{m} (\lambda^{(m)}_0 \bar{\lambda}^{(n|m)}_0) & = 2 \sqrt{\frac{m}{n}}, \\
    \mathrm{tr}_{m} (\lambda^{(m)}_0 \bar{\lambda}^{(n|m)}_{d^2-1}) & = 2 \sqrt{\frac{m}{d(d-1)}} \quad \text{if} \; d > m

where :math:`\mathrm{tr}_{m}(\cdot)` represents the :math:`m`-dimensional trace, and :math:`\bar{\lambda}^{(n|m)}_k`
is the :math:`m`-dimensional submatrix of :math:`\lambda^{(n)}_k`. Thus we have

.. math::

    \bar{\nu}_0 = \frac{1}{2} \mathrm{tr}_m (\lambda^{(m)}_0 \bar{H}^{(m)}) = \sqrt{\frac{m}{n}} \nu_0 + \sum_{d > m} \sqrt{\frac{m}{d(d-1)}} \nu_{d^2-1}.

Pauli Matrices API
==================

.. autosummary::
   :toctree: ../generated

   paulis
   components
   compose
   l0_projector
   truncate
   symmetry
   labels
"""

from typing import Sequence, Optional, Union
from types import ModuleType
import string
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    has_jax = False
else:
    has_jax = True

from ._types import array_like, ndarray, MatrixDimension

def paulis(dim: MatrixDimension) -> np.ndarray:
    r"""Return a list of generalized Pauli matrices of given dimension(s) as an array.

    Args:
        dim: Dimension(s) of the Pauli matrices.

    Returns:
        An array of Pauli (product) matrices as an array. For `dim=(d1, d2, ...)`, the shape of
        the array is `(d1**2, d2**2, ..., d1*d2*..., d1*d2*...)`.
    """
    if isinstance(dim, int):
        dim = (dim,)

    subsystems = []

    for pauli_dim in dim:
        # Compose the unnormalized matrices
        matrices = np.zeros((pauli_dim ** 2, pauli_dim, pauli_dim), dtype=complex)
        matrices[0] = np.diag(np.ones(pauli_dim))
        ip = 1
        for isub in range(1, pauli_dim):
            for irow in range(isub):
                matrices[ip, irow, isub] = 1.
                matrices[ip, isub, irow] = 1.
                ip += 1
                matrices[ip, irow, isub] = -1.j
                matrices[ip, isub, irow] = 1.j
                ip += 1

            matrices[ip, :isub + 1, :isub + 1] = np.diag(np.array([1.] * isub + [-isub]))
            ip += 1

        # Normalization
        norm = np.trace(np.matmul(matrices, matrices), axis1=1, axis2=2)
        matrices *= np.sqrt(2. / norm)[:, None, None]

        subsystems.append(matrices)

    num_sub = len(dim)

    if num_sub == 1:
        return subsystems[0]

    else:
        # Compose Pauli products
        # (d1**2, d1, d1) x (d2**2, d2, d2) -> (d1**2, d2**2, d1*d2, d1*d2)
        #      a   b   c         d   e   f          a      d     be     cf
        # be and cf are reshaped into 1 dimension each
        chars = string.ascii_letters
        if num_sub * 3 > len(chars):
            raise NotImplemented('Too many subsystems - need an implementation using recursive np.kron')

        indices_in = []
        indices_out = [''] * 3
        for il in range(0, num_sub * 3, 3):
            indices_in.append(chars[il:il + 3])
            indices_out[0] += chars[il]
            indices_out[1] += chars[il + 1]
            indices_out[2] += chars[il + 2]

        indices = f'{",".join(indices_in)}->{"".join(indices_out)}'
        dim_array = np.asarray(dim)
        shape = np.concatenate((np.square(dim_array), np.prod(np.repeat(dim_array[None, :], 2, axis=0), axis=1)))

        return np.einsum(indices, *subsystems).reshape(*shape) / (2 ** (num_sub - 1))


def components(
    matrix: array_like,
    dim: Optional[MatrixDimension] = None,
    npmod: ModuleType = np
) -> ndarray:
    r"""Return the Pauli decomposition coefficients :math:`\nu_{k_1 \dots k_n}` of the matrix.

    Args:
        matrix: Matrix to decompose. The last two dimensions of the array are dotted with the Pauli
            matrices.
        dim: Subsystem dimensions. The product of subsystem dimensions must match the matrix dimension.
            If None, the matrix is assumed to represent a single system.

    Returns:
        A complex array of shape `(..., d1**2, d2**2, ...)` where `d1`, `d2`, ... are the subsystem dimensions.

    Raises:
        ValueError: If `prod(dim)` does not match the matrix dimension.
    """
    if npmod is np:
        if dim is None:
            dim = (matrix.shape[-1],)
        elif isinstance(dim, int):
            dim = (dim,)

        if np.prod(dim) != matrix.shape[-1]:
            raise ValueError(f'Invalid subsystem dimensions {dim}')

    basis = paulis(dim)

    return npmod.tensordot(matrix, basis, ((-2, -1), (-1, -2))) * (2 ** (len(dim) - 2))


def compose(
    components: array_like,
    dim: Optional[MatrixDimension] = None,
    npmod: ModuleType = np
) -> ndarray:
    r"""Compose a matrix from the Pauli components.

    Args:
        components: Pauli components of the desired matrix, shape (..., d1**2, d2**2, ...)
        dim: Subsystem dimensions. If present, last `len(dim)` dimensions of `components`
            are dotted with the corresponding Pauli matrices.

    Returns:
        A complex array of shape `(..., d1*d2*..., d1*d2*...)`.
    """
    if npmod is np:
        if dim is None:
            dim = np.around(np.sqrt(components.shape)).astype(int)
        elif isinstance(dim, int):
            dim = (dim,)

        if not np.allclose(np.square(dim), components.shape[-len(dim):]):
            raise ValueError('Components array shape invalid')

    basis = paulis(dim)

    comp_axes = list(range(-len(dim), 0))
    pauli_axes = list(range(len(dim)))
    return npmod.tensordot(components, basis, (comp_axes, pauli_axes))


def l0_projector(reduced_dim: int, original_dim: int) -> np.ndarray:
    r"""Return the vector that projects the components of original_dim decomposition onto lambda_0 of reduced_dim.

    Args:
        reduced_dim: Matrix dimension of the target subspace.
        original_dim: Matrix dimension of the full space.

    Returns:
        Projection vector :math:`\vec{v}` that gives :math:`\bar{\nu}_0 = \vec{v} \cdot \vec{\nu}`.
    """
    if reduced_dim > original_dim:
        raise ValueError('Reduced dim greater than original dim')

    projector = np.zeros(original_dim ** 2)
    projector[0] = np.sqrt(reduced_dim / original_dim)

    for d in range(reduced_dim + 1, original_dim + 1):
        projector[d ** 2 - 1] = np.sqrt(reduced_dim / d / (d - 1))

    return projector


def truncate(
    components: array_like,
    reduced_dim: MatrixDimension,
    npmod: ModuleType = np
) -> ndarray:
    r"""Truncate a component array of a matrix into the components for a submatrix.

    The component array can have extra dimensions in front (e.g. time axis if this is a time series of components).
    In such a case, reduced_dim must be a sequence of integers with the length correpsonding to the number of
    subsystems.

    Args:
        components: Pauli components of the original matrix, shape (..., d1**2, d2**2, ...)
        reduced_dim: Dimension(s) of the submatrix(es).

    Returns:
        Components of the submatrix, shape (..., r1**2, r2**2, ...)
    """
    if npmod is np and isinstance(reduced_dim, int):
        reduced_dim = (reduced_dim,) * len(components.shape)

    num_subsystems = len(reduced_dim)
    first_component_axis = len(components.shape) - num_subsystems

    original_shape = components.shape[first_component_axis:]
    reduced_shape = np.square(reduced_dim)

    if npmod is np:
        if np.any(reduced_shape > np.asarray(original_shape)):
            raise ValueError(f'Reduced dimensions greater than original dimensions')

        if np.allclose(reduced_shape, original_shape):
            return components.copy()

    original_dim = npmod.around(npmod.sqrt(original_shape)).astype(int)

    def project_dim(idim, components):
        # Construct a matrix (v, diag([1] * reduced)[1:], diag([0] * truncated))^T
        projector = l0_projector(reduced_dim[idim], original_dim[idim])
        diag = npmod.concatenate((npmod.ones(reduced_shape[idim]),
                                  npmod.zeros(original_shape[idim] - reduced_shape[idim])))
        projector = npmod.concatenate((projector[None, :], npmod.diag(diag)[1:]), axis=0)

        projected = npmod.tensordot(projector, components, (1, first_component_axis + idim))

        # After tensordot, the projected axis is at position 0
        transpose = list(range(1, len(components.shape)))
        transpose.insert(first_component_axis + idim, 0)

        return projected.transpose(transpose)

    if has_jax and npmod is jnp:
        def loop_body(idim, components):
            return jax.lax.cond(reduced_dim[idim] == original_dim[idim],
                                lambda c: c,
                                lambda c: project_dim(idim, c),
                                components)

        components = jax.lax.fori_loop(0, num_subsystems, loop_body, components)

    else:
        for idim in range(num_subsystems):
            if reduced_dim[idim] != original_dim[idim]:
                components = project_dim(idim, components)

    slices = tuple(slice(shape) for shape in reduced_shape)

    return components[(...,) + slices]


def symmetry(dim: int):
    r"""Return the symmetry (-1, 0, 1) of the Pauli matrices.

    Args:
        dim: Dimension of the Pauli matrices.

    Returns:
        An 1D integer array with entries -1, 0, 1 depending on whether the corresponding Pauli matrix
        is antisymmetric, diagonal, or symmetric.
    """

    symmetry = np.zeros(dim ** 2, dtype=int)

    ip = 1
    for isub in range(1, dim):
        for irow in range(isub):
            symmetry[ip] = 1
            ip += 1
            symmetry[ip] = -1
            ip += 1

        ip += 1

    return symmetry


def labels(
    dim: MatrixDimension,
    symbol: Optional[Union[str, Sequence[str]]] = None,
    delimiter: str = '',
    norm: bool = True,
    fmt: str = 'latex'
) -> np.ndarray:
    r"""Generate the labels for the Pauli matrices of a given dimension.

    Args:
        dim: Dimension(s) of the Pauli matrices.
        symbol: Base symbol.
        delimiter: Delimiter between the symbols for multibody labels.
        norm: Include the normalization factors.
        fmt: Output format. Allowed values are 'text', 'latex', 'latex-text',
            'latex-slash'.

    Returns:
        An ndarray of type string and shape `(d1**2, d2**2, ...)`.
    """
    if isinstance(dim, int):
        dim = (dim,)

    if symbol is None or isinstance(symbol, str):
        symbol = (symbol,) * len(dim)

    out = np.array('', dtype=str)

    for pauli_dim, sym in zip(dim, symbol):
        if delimiter and len(out.shape) > 0:
            out = np.char.add(out, np.full(out.shape, delimiter))

        if not sym:
            if pauli_dim == 2:
                labels = ['I', 'X', 'Y', 'Z']
            elif sym is None:
                if fmt == 'text':
                    labels = list(f'λ{i}' for i in range(pauli_dim ** 2))
                else:
                    labels = list(fr'\lambda_{{{i}}}' for i in range(pauli_dim ** 2))
            else:
                labels = list(str(i) for i in range(pauli_dim ** 2))
        else:
            labels = list(f'{sym}_{{{i}}}' for i in range(pauli_dim ** 2))

        out = np.char.add(np.repeat(out[..., None], pauli_dim ** 2, axis=-1), labels)

    if norm and len(dim) >= 2:
        if len(dim) == 2:
            denom = '2'
        elif fmt == 'text':
            denom = f'2**{len(dim) - 1}'
        else:
            denom = '2^{%d}' % (len(dim) - 1)

        if fmt in ('text', 'latex-slash'):
            post = np.full(out.shape, f'/{denom}')

        else:
            if fmt == 'latex':
                pre = np.full(out.shape, r'\frac{')
                post = np.full(out.shape, '}{%s}' % denom)
            else:
                pre = np.full(out.shape, r'\textstyle{\frac{')
                post = np.full(out.shape, '}{%s}}' % denom)

            out = np.char.add(pre, out)

        out = np.char.add(out, post)

    return out
