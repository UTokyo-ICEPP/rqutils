r"""
======================================================================
Sample-based quantum diagonalization of general Pauli-sum Hamiltonians
======================================================================

.. currentmodule:: rqutils.sqd

Overview
========

SQD is an algorithm for finding an approximate ground eigenpair of a very large (computationally
intractable) Hamiltonian by projecting it onto a subspace. In our case, the Hamiltonian is expressed
as a linear combination of Pauli strings, and the subspace is identified from a (possibly redundant)
set of bitstrings (states).

Typically, the projected Hamiltonian is itself still too large to be stored in memory as a matrix,
requiring some matrix-free method to solve the eigenvalue problem. The technical challenge is then
threefold:

- Extracting the set :math:`S` of unique bitstrings from the input list.
- Computing (on the fly) the matrix elements :math:`\langle j | Q | k \rangle` for all
  :math:`j, k \in S` for each term :math:`Q` of the Hamiltonian.
- Solving the eigenvalue problem.

The first point will have to be done with ``np.unique`` or an equivalent accelerated function. For
the last point, we can use the `ground_locg` solver provided in this package, which takes a
matrix-vector application function and an initial-guess vector as inputs. The central task here is
thus providing the matvec function that covers the second point.

Usage examples can be found at
`examples/sqd.py <https://github.com/UTokyo-ICEPP/rqutils/tree/main/examples/sqd.py>`__.

Algorithm
=========

The algorithm takes advantage of the symplectic representation of Pauli strings. A term in the
Hamiltonian is a product of a real coefficient :math:`\alpha` and a Pauli string :math:`Q`. In the
symplectic representation,

.. math::

    \alpha Q = \alpha (-i)^{xz} \left(Z^{z_{n-1}} \otimes \cdots Z^{z_{0}}\right)
                                  \left(X^{x_{n-1}} \otimes \cdots X^{x_{0}}\right)

where :math:`x` (X signature) and :math:`z` (Z signature) are binary vectors of length :math:`n`
(number of qubits) and :math:`xz` represents their inner product. For a given bitstring
:math:`s = [s_{n-1}, \dots, s_{0}]`, the X signature of the Hamiltonian term determines the
existence and location of the matrix element, and the Z signature gives the sign.

In the preparation stage of the algorithm, the terms of the input Hamiltonian are grouped by the X
signature (for example, XIZ and YZI will belong to the same group). For each X signature, there will
be multiple Z signatures and the corresponding phased coefficients (:math:`\alpha (-i)^{xz}` above).
The X and Z signatures are bit-packed into arrays of 8-bit integers.

The input states are then sorted and similarly bit-packed to allow bitwise operations between the
states and the X/Z signatures. The resulting array :math:`S = [s^{0}, \dots, s^{N-1}]` is the basis
on which the Hamiltonian is projected. We then define the initial one-hot vector of length :math:`N`
as the input to the LOBPCG function.

Let :math:`J` be the number of distinct X signatures in the Hamiltonian, and :math:`K^{(j)}` be the
number of Z signatures and coefficients associated with the :math:`j` th X signature. The
matrix-vector operation to be passed to the solver acts on the length-:math:`N` vector :math:`v` of
coefficients as

.. math::

    v' = \sum_{j=1}^{J} \left( \sum_{k=1}^{K^{(j)}} \alpha^{(j,k)} (-i)^{x^{(j)}z^{(j,k)}}
                                            D[z^{(j,k)}] \right) \circ B[x^{(j)}](v).

The operation

.. math::

    w = B[x](v)

consists of the following steps:

#. Compute the source state :math:`t^{i} \leftarrow s^{i} \oplus x` of :math:`w^{i}`.
#. If a source index :math:`j^{i}` exists such that :math:`s^{j^{i}} = t^{i}`,
   :math:`w^{i} \leftarrow v^{j^{i}}`. Otherwise :math:`w^{i} \leftarrow 0`.

The operation :math:`D[z]` is a diagonal operation that applies a sign factor to each vector entry:

.. math::

    D[z](w^{i}) = (-1)^{zs^{i}} w^{i}.

Caching
-------

In the expressions above, source indices :math:`[j^{i}]` and sign factors :math:`[(-1)^{zs^{i}}]` do
not depend on the coefficient vector :math:`v` and can be determined once :math:`S` is given. In
fact, the composition of the sign factors with the coefficients

.. math::

    C^{(j)} = \sum_{k=1}^{K^{(j)}} \alpha^{(j,k)} (-i)^{x^{(j)}z^{(j,k)}} [(-1)^{z^{(j,k)}s^{i}}]

is entirely static in the same way. It is therefore possible to consider caching these vectors and
reusing them in the repeated call to the matrix-vector function. There is however a tradeoff between
the compute time and memory footprint, as is always the case with caching.

Concretely, caching the source indices :math:`[j^{i}]` requires :math:`4 J N` bytes of memory,
assuming :math:`N \leq 2^{31}` (which is actually the hard limit set by other constraints; see the
next section) and therefore 32-bit (4-byte) integers are used for vector indexing. Caching the sign
bits will require :math:`\kappa N` bytes, where

.. math::

    \kappa = \left\lceil \frac{\sum_{j} K^{(j)}}{8} \right\rceil

is the number of bytes required to pack the sign bits for each vector entry. This "dense" packing
however may cause inefficiencies in computation that defeats the purpose of caching. For a more
straightforward caching, the memory requirement is inflated to :math:`\bar{\kappa} J N` bytes, where

.. math::

    \bar{\kappa} = \left\lceil \frac{\max_{j} K^{(j)}}{8} \right\rceil.

If the entire composition of the coefficients are cached instead of the sign bits, :math:`8 J N` or
:math:`16 J N` bytes are used, depending on whether the vector is real or complex (i.e., if there
are terms in the Hamiltonian with odd numbers of Ys).

Given that identifying :math:`[j^{i}]` is an expensive operation, while the other diagonal
operations are not, the default behavior of this function is to cache only the source indices.
Note however that there are cases when further caching can actually be advantageous also in terms of
memory. This is because :math:`S` will not be used after caching both the source indices and sign
bits / coefficient sums. Since :math:`S` occupies :math:`\left\lceil n/8 \right\rceil N` bytes of
memory, caching setting should be adjusted according to the values of :math:`n` and
:math:`\{K^{(j)}\}_j`.

Distributed arrays and scaling limits
=====================================

When the SQD function is called within a context where the global mesh is set via
``jax.set_mesh(mesh)``, the state vector is distributed (sharded) among the devices in the mesh,
and accordingly all arrays with an axis with size :math:`N` follow the same sharding. Even the most
aggressive caching strategy described above will be possible this way.

However, there is a limit to scaling in :math:`N` (SQD subspace dimension) imposed by the need to
sort the states list during the initial uniquification, and also whenever the source indices for an
X signature is computed. At the moment, sorting must take place within a single device, with at most
:math:`2^{32}` elements involved. Furthermore, source indices identification sorts through a stack of
two state lists. Therefore, the maximum achievable :math:`N` is :math:`2^{31}`. A comparable limit
is set by the GPU memory, which is at most O(100)GB per device as of mid-2026.

When the source indices are cached but neither the sign bits nor the diagonals are, the state list
:math:`S` will also be sharded after the computation of the source indices are done.

SQD API
=======

.. autofunction:: sqd
.. autofunction:: hproj
"""
from collections.abc import Sequence
import logging
import time
from numbers import Number
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, coo_array
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, get_abstract_mesh
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.ground_locg import ground_locg

LOG = logging.getLogger(__name__)

type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]]
try:
    from qiskit.quantum_info import SparsePauliOp
    HamiltonianInput |= SparsePauliOp
except ImportError:
    pass
type Vector = np.ndarray[tuple[int], np.dtype[np.inexact]]
type StateList = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type PackedStateList = np.ndarray[tuple[int, int], np.dtype[np.uint8]]

@dataclass(frozen=True)
class SQDOptions:
    return_eigvec: bool = True
    """Whether to return the eigenvector (coefficients and unique state bitstrings)."""
    cache_strategy: tuple[int, int] = (1, 0)
    """Switches for caching the results of source indices and sign bits / diagonals.
       See the module documentation for the detailed discussion of the resource tradeoff involved."""
    gather_shard_bundle: int = 1
    """How many shards to bundle in the staged cache. Only used with cache_strategy (2, *)."""
    locg_maxiter: int = 1000
    """LOCG maximum iterations."""
    locg_tol: float | None = None
    """LOCG convergence tolerance."""


def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    states_size: Optional[int] = None,
    options: Optional[SQDOptions | dict[str, Any]] = None
) -> float | tuple[float, Vector, StateList]:
    r"""Perform a sample-based quantum diagonalization of the Hamiltonian.

    The Hamiltonian can be given in three different forms:
    
    * A tuple of two lists, where the first list enumerates the Pauli strings :math:`Q` as strs and
      the second contains the coefficients :math:`\alpha`.
    * Qiskit SparsePauliOp
    * PauliSumXZ (From :mod:`rqutils.paulis.symplectic`)

    States must have binary values and can be passed as an array of integers or booleans.

    Internally, the states are bit-packed and represented by :math:`\lceil (n+1)/8 \rceil`
    ``uint8`` s, where the extra bit is placed at position 0 and serves as the indicator for
    spurious (fill-in) entries. Accordingly, the PauliSumXZ representation of the Hamiltonian must
    be made with ``add_padding=True``.

    Cache level is a 2-tuple where the first element specifies the caching of the source indices
    (0=no caching, 1=cached) and the second specifies the caching of the diagonal elements (0=no
    caching, 1=cache sign bits, 2=cache diagonals).

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits).
        states_size: Fix the size of the states array used in computation to the specified value so
            that compilation is not triggered at each call with slightly different array sizes.
        options: Execution options.

    Returns:
        Calculated ground state energy, or a tuple of energy, ground state vector, and sorted
        uniquified states (if return_eigvec=True).
    """
    if isinstance(options, dict):
        options = SQDOptions(**options)

    if states_size is None:
        states_size = states.shape[0]
    if states_size < states.shape[0]:
        raise ValueError('states_size smaller than the states array length')
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian, force_real=True, add_padding=True)

    if not (mesh := get_abstract_mesh()).empty and (resid := states_size % mesh.size) != 0:
        LOG.debug('Adjusting states_size to make the array divisible by %d', mesh.size)
        states_size += mesh.size - resid

    # Need an extra left zero bit to distinguish fill-in states from the inputs after
    # unique(..., size=X, fill_value=255)
    # We need to insert the bit with pad() at this point because packbits() fills from the left
    # Perhaps write a new ufunc if this becomes too slow for very large input?
    states_p = np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)

    LOG.debug('Starting SQD with array size %s', states_size)
    start = time.time()
    result = run_sqd(hamiltonian, states_p, states_size, options)
    LOG.info('Found ground eigenpair in %f seconds.', time.time() - start)
    eigval = float(result[0])
    if options.return_eigvec:
        eigvec, states_u, subspace_dim = result[1:]
        basis_states = np.unpackbits(states_u[:subspace_dim], axis=-1)[:, 1:1 + states.shape[1]]
        return (eigval, np.array(eigvec[:subspace_dim]), basis_states)
    return eigval


def hproj(
    hamiltonian: HamiltonianInput,
    states: StateList,
    unique_states: bool = False
) -> csr_array:
    """Return the Hamiltonian projected onto the given subspace.

    The Hamiltonian can be given in three different forms:
    
    * A tuple of two lists, where the first list enumerates the Pauli strings :math:`Q` as strs and
      the second contains the coefficients :math:`\alpha`.
    * Qiskit SparsePauliOp
    * PauliSumXZ (From :mod:`rqutils.paulis.symplectic`)

    States must have binary values and can be passed as an array of integers or booleans.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits).
        unique_states: Whether the states can be assumed to be already uniquified and sorted.

    Returns:
        The projected Hamiltonian as a sparse matrix.
    """
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian, add_padding=True)
    if not unique_states:
        states = np.unique(states, axis=0)
    states_p = np.packbits(states, axis=1)

    @jax.jit
    def cols_elems(_hamiltonian, _states_p):
        def get_from_one(_, ham):
            columns = get_xsource(ham[0], _states_p)
            diagonals = get_diagonal(ham[1], ham[2], _states_p)
            return None, (columns, diagonals)

        return jax.lax.scan(get_from_one, None, _hamiltonian.arrays)[1]

    columns, elements = cols_elems(hamiltonian, states_p)
    valid = columns != -1
    rows = np.tile(np.arange(states.shape[0])[None, :], (columns.shape[0], 1))[valid]
    data = np.array(elements[valid])
    cols = np.array(columns[valid])
    return csr_array(coo_array((data, (rows, cols))))


@jax.jit(
    static_argnames=[
        'states_size',
        'options',
        'log_level'
    ]
)
def run_sqd(
    hamiltonian: PauliSumXZ,
    states_p: PackedStateList,
    states_size: int,
    options: SQDOptions | None = None,
    log_level: int = logging.INFO
) -> tuple[float] | tuple[float, jax.Array, jax.Array, int]:
    """JIT-compiled part of the SQD function."""
    if (mesh := get_abstract_mesh()).empty:
        raise RuntimeError('Context mesh is required. Use jax.set_mesh() to set a mesh.')
    axis_names = mesh.axis_names

    if options is None:
        options = SQDOptions()

    if log_level <= logging.DEBUG:
        jax.debug.print(ordered=True)('Uniquifying states (size {})', states_size)

    states_u = uniquify_states(states_p, states_size)

    arg_parts = {
        'x': (hamiltonian.x, P(None, None)),
        'z': (hamiltonian.z, P(None, None, None)),
        'c': (hamiltonian.c, P(None, None)),
        'states_u': (states_u, P(None, None))
    }

    if options.cache_strategy[0] > 0:
        if log_level <= logging.DEBUG:
            jax.debug.print(ordered=True)('Precomputing xsources')

        xsources = jax.lax.scan(
            lambda _, x: (None, get_xsource(x, states_u)),
            None,
            hamiltonian.x
        )[1]
        if mesh.size > 1:
            # We will not be performing sorts on states any more - shard the array
            if log_level <= logging.DEBUG:
                jax.debug.print(ordered=True)('Sharding states array')
            states_u = jax.reshard(states_u, P(axis_names, None))
            arg_parts['states_u'] = (states_u, P(axis_names, None))

    if options.cache_strategy[1] == 1:
        if log_level <= logging.DEBUG:
            jax.debug.print(ordered=True)('Precomputing sign bits of diagonals')

        diag_signs = jax.lax.scan(
            lambda _, z: (None, get_diag_signs(z, states_u)),
            None,
            hamiltonian.z
        )[1]
        arg_parts['diag_signs'] = (diag_signs, P(None, axis_names, None))

    elif options.cache_strategy[1] == 2:
        if log_level <= logging.DEBUG:
            jax.debug.print(ordered=True)('Precomputing diagonals')

        diagonals = jax.lax.scan(
            lambda _, v: (None, get_diagonal(v[0], v[1], states_u)),
            None,
            (hamiltonian.z, hamiltonian.c)
        )[1]
        arg_parts['diagonals'] = (diagonals, P(None, axis_names))

    match options.cache_strategy:
        case (0, 0):
            matvec = apply_h
            args = (hamiltonian.x, hamiltonian.z, hamiltonian.c, states_u)
        case (0, 1):
            matvec = apply_h_s_cached
            args = (hamiltonian.x, states_u, diag_signs, hamiltonian.c)
        case (0, 2):
            matvec = apply_h_z_cached
            args = (hamiltonian.x, states_u, diagonals)
        case (1, 0):
            matvec = apply_h_x_cached
            args = (xsources, hamiltonian.z, hamiltonian.c, states_u)
        case (1, 1):
            matvec = apply_h_xs_cached
            args = (xsources, diag_signs, hamiltonian.c)
        case (1, 2):
            matvec = apply_h_xz_cached
            args = (xsources, diagonals)
        case (2, 0):
            in_specs=(
                P(None, axis_names),
                P(None, None, None),
                P(None, None),
                P(axis_names, None)
            )
            matvec = apply_h_x_cached_by_shard
            args = (xsources, hamiltonian.z, hamiltonian.c, states_u)
            in_specs = (P(axis_names),) + in_specs
            matvec = jax.jit(jax.shard_map(matvec, in_specs=in_specs, out_specs=P(axis_names)))

    def vinit_from_min_diag():
        if options.cache_strategy[1] == 2:
            diagonal = diagonals[0]
        else:
            diagonal = get_diagonal(hamiltonian.z[0], hamiltonian.c[0], states_u).real
        # Set the fill-in components to the maximum value so that argmin only sees the valid entries
        diagonal = jnp.where(states_u[:, 0] == 255, np.inf, diagonal)
        return jnp.argmin(diagonal)

    def vinit_nodiag():
        # TODO: Come up with a way to find a good vinit when there is no diagonal term
        return 0

    if log_level <= logging.DEBUG:
        jax.debug.print(ordered=True)('Generating vinit')

    vinit = jax.lax.cond(
        jnp.all(hamiltonian.x[0] == 0),
        vinit_from_min_diag,
        vinit_nodiag
    )

    if log_level <= logging.DEBUG:
        jax.debug.print(ordered=True)('Starting minimization with cache_strategy {}',
                                      options.cache_strategy)

    eigval, eigvec, _ = ground_locg(matvec, vinit, args=args, maxiter=options.locg_maxiter,
                                    tol=options.locg_tol, vspace=(states_size, hamiltonian.c.dtype),
                                    log_level=log_level)
    result = (eigval,)
    if options.return_eigvec:
        if mesh.size > 1:
            eigvec = jax.reshard(eigvec, P(None))
            states_u = jax.reshard(states_u, P(None, None))
        subspace_dim = jnp.searchsorted(states_u[:, 0] >> 7, 1)
        result += (eigvec, states_u, subspace_dim)
    return result


@jax.jit(static_argnames=['states_size', 'return_indices'])
def uniquify_states(
    states_p: PackedStateList,
    states_size: int,
    return_indices: bool = False
) -> StateList:
    """A stripped-down implementation of jnp.unique.

    The returned array will have shape (states_size, states_p.shape[1]). If states_size is greater
    than the number of unique states, the residual entries at the end are filled with 255.
    """
    # Perform a lexsort
    iota = jax.lax.broadcasted_iota(np.int32, (states_p.shape[0],), 0)
    perm = jax.lax.sort((*states_p.T, iota), dimension=0, num_keys=states_p.shape[1])[-1]
    states_srt = states_p[perm]
    # Uniqueness flag for elements 1 to N-1
    is_unique = jnp.any(jax.lax.ne(states_srt[1:], states_srt[:-1]), axis=1)
    # Element 0 is always considered unique -> add 1
    total_unique = jnp.sum(is_unique, dtype=np.int32) + 1
    # This cumsum(bincount(cumsum)) accounts for the uniqueness of the 0th element
    idx_unique = jnp.cumsum(
        jnp.bincount(
            jnp.cumsum(is_unique, dtype=np.int32),
            length=states_size
        ),
        dtype=np.int32
    )
    # Finally flag out filler slots by setting total_unique: to -1
    if states_size != states_p.shape[0]:
        iota = jax.lax.broadcasted_iota(np.int32, (states_size,), 0)
    idx_unique = jnp.where(iota < total_unique, idx_unique, -1)
    # With wrap_negative_indices=False we'll have 255 for filler slots
    uniquified = states_srt.at[idx_unique].get(mode='fill', fill_value=255,
                                               wrap_negative_indices=False)
    if return_indices:
        return uniquified, perm.at[idx_unique].get(mode='fill', fill_value=-1,
                                                   wrap_negative_indices=False)
    return uniquified


@jax.jit(static_argnames=['sorted_input'])
def get_xsource(
    xsignature: NDArray[np.uint8],
    states_p: PackedStateList,
    sorted_input: bool = True
) -> jax.Array:
    """Return an index array into the source of an X operation.

    Let `V` be a vector of complex or float values with shape `[N]`, `S` be a unique 2-d array
    of uint8 with shape `[N, B]` where `B = ceil(Q/8)`, and `X` be a vector of uint8 with shape
    `[B]`. An unpacked (truncated to `Q` bits) `X` is a bitstring that represents the location of X
    being applied to the states in `S`; X (I) is applied to qubit `q` if `Q-q-1`th bit is 1 (0). Let
    `P` be the projector of the shape `[2 ** Q]` state vector `W` onto `V`.

    We want to find a vector of indices `A` where `(PXW)[i] = V[A[i]]`. This is trivial if `P` is
    the identity (or equivalently if `S` contains all bitstrings from `0` to `2^Q-1`), because then
    we know that `S[i] = i` and therefore `A[i] = i ^ X`. In the presence of a nontrivial
    projection, we must take care of not only the source location but also the existence of the
    source itself, since it is not guaranteed that `S[i] ^ X` is in `S`. When it is not, we set
    `A[i] = -1` so that `V[A[i]]` can default to a `fill_value` of 0.0 through `at[].get()` applied
    to `V`.

    To find `A`, we first concatenate `S` and `S ^ X` into a `[2N, B]` array and perform a stable
    sort along axis 0 to obtain an array `T`. Indices from `0` to `2N` are sorted together so that
    the resulting index array `I` has value `i` at index `k` such that `T[k] = S[i]` (`i < N`) or
    `T[k] = (S^X)[i-N]` (`i >= N`). Then, if `T[k] == T[k+1]`, `A[I[k]] = I[k+1] - N`. On the other
    hand, `T[k] != T[k+1]` where `I[k] < N` implies that the source bitstring does not exist for
    `S[I[k]]` and therefore `A[I[k]]` must be set to `-1`.
    """
    size = states_p.shape[0]
    if not sorted_input:
        states_p, sort_indices = uniquify_states(states_p, size, return_indices=True)
    mapped_states = jnp.bitwise_xor(states_p, xsignature)  # S^X
    joined = jnp.concatenate([states_p, mapped_states], axis=0)
    idx = jax.lax.iota(np.int32, 2 * size)
    # lax.sort seems to leak GPU memory; can lose as much as 5 GB when sorting x of shape (5M,9)
    sorted = jax.lax.sort(tuple(joined.T) + (idx,), num_keys=joined.shape[1])
    joined_sorted = jnp.stack(sorted[:-1], axis=1)  # T
    idx_sorted = sorted[-1]  # I
    invalid = np.array(-1, dtype=np.int32)

    source_idx = jnp.where(
        jnp.all(jnp.equal(joined_sorted[:-1], joined_sorted[1:]), axis=1),  # T[k] == T[k+1]
        idx_sorted[1:] - size,  # I[k+1] - N
        invalid
    )
    # Stripped-down jnp.nonzero implementation (with dtype control; othersize int64 is used
    # unnecessarily)
    tposition = jnp.cumsum(
        jnp.bincount(
            jnp.cumsum(idx_sorted < size, dtype=np.int32),
            length=size
        ),
        dtype=np.int32
    )
    xsource = source_idx.at[tposition].get(mode='fill', fill_value=invalid)
    if not sorted_input:
        xsource = jnp.arange(size).at[sort_indices].set(
            sort_indices.at[xsource].get(mode='fill', fill_value=invalid,
                                         wrap_negative_indices=False)
        )
    if not (mesh := get_abstract_mesh()).empty:
        xsource = jax.reshard(xsource, P(mesh.axis_names))
    return xsource


@jax.jit
def get_diag_signs(
    zsignatures: NDArray[np.uint8],
    states: StateList
) -> jax.Array:
    """Return the packed sign bits."""
    out_sharding = jax.typeof(states).sharding

    def get_signs(carry, zsignature):
        out, ibyte, ibit = carry
        sign_bits = jnp.sum(jnp.bitwise_count(states & zsignature), axis=1, dtype=np.uint8) & 1
        # bits and bytes are counted from the left
        out = out.at[:, ibyte].add(sign_bits << (7 - ibit), out_sharding=out_sharding)
        ibyte, ibit = jax.lax.cond(
            ibit == 7,
            lambda: (ibyte + 1, 0),
            lambda: (ibyte, ibit + 1)
        )
        return (out, ibyte, ibit), None

    num_bytes = np.ceil(zsignatures.shape[0] / 8).astype(int)
    init = jnp.zeros((states.shape[0], num_bytes), dtype=np.uint8, out_sharding=out_sharding)
    return jax.lax.scan(get_signs, (init, 0, 0), zsignatures)[0][0]


@jax.jit
def compute_diagonal(
    diag_signs: NDArray[np.uint8],
    coeffs: NDArray[np.inexact]
) -> jax.Array:
    """Compute the diagonals from the sign bits and coefficients."""
    def cond_fn(val):
        iterm = val[1]
        return jnp.logical_and(iterm < coeffs.shape[0], jnp.not_equal(coeffs[iterm], 0.))

    def add_diag(val):
        diagonal, iterm = val
        coeff = coeffs[iterm]
        ibyte = iterm // 8
        ibit = iterm & 255
        signed = (diag_signs[:, ibyte] >> (7 - ibit)) & 1
        signs = 1. - 2. * signed
        return diagonal + coeff * signs, iterm + 1

    signs_sharding = jax.typeof(diag_signs).sharding
    out_sharding = NamedSharding(signs_sharding.mesh, P(signs_sharding.spec[0]))
    init = jnp.zeros(diag_signs.shape[0], dtype=coeffs.dtype, out_sharding=out_sharding)
    return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]


@jax.jit
def get_diagonal(
    zsignatures: NDArray[np.uint8],
    coeffs: NDArray[np.inexact],
    states: StateList
) -> jax.Array:
    """Return the fully composed diagonals for one X signature."""
    # Null terms are removed with hamiltonian.simplify() so we iterate until we hit coeff=0
    def cond_fn(val):
        iterm = val[1]
        return jnp.logical_and(iterm < coeffs.shape[0], jnp.not_equal(coeffs[iterm], 0.))

    def add_diag(val):
        diagonal, iterm = val
        zsignature = zsignatures[iterm]
        coeff = coeffs[iterm]
        signed = jnp.sum(jnp.bitwise_count(states & zsignature), axis=1, dtype=np.uint8) & 1
        signs = 1. - 2. * signed
        return diagonal + coeff * signs, iterm + 1

    states_sharding = jax.typeof(states).sharding
    out_sharding = NamedSharding(states_sharding.mesh, P(states_sharding.spec[0]))
    init = jnp.zeros(states.shape[0], dtype=coeffs.dtype, out_sharding=out_sharding)
    if (varying_axes := jax.typeof(states).manual_axis_type.varying):
        init = jax.lax.pcast(init, tuple(varying_axes), to='varying')
    return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]


@jax.jit
def apply_xgrp(
    xsource: NDArray[np.int32],
    diagonal: NDArray[np.inexact],
    vec: NDArray[np.inexact]
) -> jax.Array:
    """Gather vector entries from the source indices and multiply them with diagonals."""
    xvec = vec.at[..., xsource].get(mode='fill', fill_value=0., wrap_negative_indices=False,
                                    out_sharding=jax.typeof(vec).sharding)
    return xvec * diagonal


@jax.jit
def apply_h(
    vec: NDArray[np.inexact],
    xsignatures: NDArray[np.uint8],
    zsignatures: NDArray[np.uint8],
    coeffs: NDArray[np.inexact],
    states: StateList
) -> jax.Array:
    """Return Hv using X and Z signatures and Pauli coefficients."""
    def fn(out, val):
        xsig, zsigs, cs = val
        xsource = get_xsource(xsig, states)
        diagonal = get_diagonal(zsigs, cs, states)
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), (xsignatures, zsignatures, coeffs))[0]


@jax.jit
def apply_h_s_cached(
    vec: NDArray[np.inexact],
    xsignatures: NDArray[np.uint8],
    states: StateList,
    diag_signs: NDArray[np.uint8],
    coeffs: NDArray[np.inexact]
) -> jax.Array:
    def fn(out, val):
        xsig, signs, cs = val
        xsource = get_xsource(xsig, states)
        diagonal = compute_diagonal(signs, cs)
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), (xsignatures, diag_signs, coeffs))[0]


@jax.jit
def apply_h_z_cached(
    vec: NDArray[np.inexact],
    xsignatures: NDArray[np.uint8],
    states: StateList,
    diagonals: NDArray[np.inexact]
) -> jax.Array:
    """Return Hv using precomputed xsources and diagonals data."""
    def fn(out, val):
        xsource = get_xsource(val[0], states)
        return out + apply_xgrp(xsource, val[1], vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), (xsignatures, diagonals))[0]


@jax.jit
def apply_h_x_cached(
    vec: NDArray[np.inexact],
    xsources: NDArray[np.int32],
    zsignatures: NDArray[np.uint8],
    coeffs: NDArray[np.inexact],
    states: StateList
) -> jax.Array:
    """Return Hv using precomputed xsources and diagonals data."""
    def fn(out, val):
        xsource, zsigs, cs = val
        diagonal = get_diagonal(zsigs, cs, states)
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), (xsources, zsignatures, coeffs))[0]


@jax.jit
def apply_h_xs_cached(
    vec: NDArray[np.inexact],
    xsources: NDArray[np.int32],
    diag_signs: NDArray[np.uint8],
    coeffs: NDArray[np.inexact]
) -> jax.Array:
    """Return Hv using precomputed xsources and diagonals data."""
    def fn(out, val):
        xsource, dsigns, cs = val
        diagonal = compute_diagonal(dsigns, cs)
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), (xsources, diag_signs, coeffs))[0]


@jax.jit
def apply_h_xz_cached(
    vec: NDArray[np.inexact],
    xsources: NDArray[np.int32],
    diagonals: NDArray[np.inexact]
) -> jax.Array:
    """Return Hv using precomputed xsources and diagonals data."""
    return jax.lax.scan(
        lambda out, val: (out + apply_xgrp(val[0], val[1], vec), None),
        jnp.zeros_like(vec),
        (xsources, diagonals)
    )[0]


def apply_h_x_cached_by_shard(
    vec_block: NDArray[np.inexact],
    xsource_blocks: NDArray[np.int32],
    zsignatures: NDArray[np.uint8],
    coeffs: NDArray[np.inexact],
    states_block: StateList
) -> NDArray[np.inexact]:
    mesh = get_abstract_mesh()
    my_idx = jax.lax.axis_index(mesh.axis_names)

    def apply_h_from_shard(shard_idx, out_block):
        in_vec = jax.lax.psum(vec_block * (shard_idx == my_idx), mesh.axis_names)
        offset = vec_block.shape[0] * shard_idx

        def fn(out, val):
            xsource_block, zsigs, cs = val
            diagonal_block = get_diagonal(zsigs, cs, states_block)
            return out + apply_xgrp(xsource_block - offset, diagonal_block, in_vec), None
    
        out_block = jax.lax.scan(
            fn,
            out_block,
            (xsource_blocks, zsignatures, coeffs)
        )[0]
        return out_block

    return jax.lax.fori_loop(
        0,
        mesh.size,
        apply_h_from_shard,
        jnp.zeros_like(vec_block)
    )


# def make_apply_h_x_gathered(shard_bundle: int):
#     mesh = get_abstract_mesh()
#     num_bundles = mesh.size // shard_bundle

#     @jax.jit(static_argnames=['ibundle'])
#     def stage_bundle(vec, ibundle):
#         shard_size = vec.shape[-1] // mesh.size
#         bundle_size = shard_size * shard_bundle
#         start = bundle_size * ibundle
#         # return vec.at[start:start + bundle_size].get(out_sharding=P())
#         bundle = auto_axes(lambda a: a[start:start + bundle_size], axes=mesh.axis_names,
#                            out_sharding=P())(vec)
#         return bundle
    
#     @jax.jit
#     def apply_h_x_gathered(
#         vec: NDArray[np.inexact],
#         xsources: NDArray[np.int32],
#         zsignatures: NDArray[np.uint8],
#         coeffs: NDArray[np.inexact],
#         states: StateList
#     ) -> jax.Array:
#         """Return Hv using precomputed xsources and diagonals data."""
#         def iter_xsig(carry, val):
#             out, bundle = carry
#             xsource, zsigs, cs = val
#             diagonal = get_diagonal(zsigs, cs, states)
#             return (out + apply_xgrp(xsource, diagonal, bundle), bundle), None

#         out = jnp.zeros_like(vec)
#         for ibundle in range(num_bundles):
#             bundle = stage_bundle(vec, ibundle)
#             offset = bundle.shape[0] * ibundle
#             # Perform gather and *diagonal
#             out = jax.lax.scan(iter_xsig, (out, bundle),
#                                (xsources - offset, zsignatures, coeffs))[0][0]

#         return out

#     return apply_h_x_gathered
