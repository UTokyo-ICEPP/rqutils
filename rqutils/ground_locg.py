r"""
==================================================
Single-vector LOBPCG. (:mod:`rqutils.ground_locg`)
==================================================

.. currentmodule:: rqutils.ground_locg

Overview
========

This module defines a single function ``ground_locg``, which implements a single-vector version
of the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) solver[1]. The structure
heavily borrows from ``jax.experimental.sparse.linalg.lobpcg_standard``, with optimizations for
single-vector (ground eigenpair) calculation.

LOBPCG is a matrix-free method for finding the extremal eigenpairs of a generalized eigenvalue
problem

.. math::

    A x = \lambda B x

with :math:`(A, B)` Hermitian. Our implementation only solves the non-generalized (:math:`B = I`)
problem and finds the minimum eigenvalue :math:`\lambda_0` and the corresponding eigenvector
:math:`v_0`.

The basic arguments to the function are

- Matrix :math:`A`, either as a JAX Array or a function that takes the vector :math:`x` (as a JAX
  Array) as input and returns :math:`Ax`.
- Initial vector, which must have a non-vanishing overlap with :math:`v_0`.

Algorithm
=========

See reference [1] for details. The goal is to minimize the Rayleigh quotient

.. math::

    \{\lambda_0, v_0\} = \{\min_{x}, \mathrm{argmin}_{x}\}
    \left(\rho (x) := \frac{x^{\dagger} A x}{|x|^2} \right) .


Conceptually, the algorithm consists of gradient descent iterations

.. math::

    x_{i+1} = x_{i} + \alpha_{i} r_{i},

where :math:`r_{i} := A x_{i} - \rho (x_{i}) x_{i}` can be proven to be proportional to the gradient
of :math:`\rho (x_{i})`. Note that :math:`r_{i}` is orthogonal to :math:`x_{i}`:

.. math::

    x_{i}^{\dagger} r_{i} & = x_{i}^{\dagger} A x_{i}
                              - \frac{x_{i}^{\dagger} A x_{i}}{|x_{i}|^2} x_{i}^{\dagger} x_{i} \\
                          & = 0.


In practice, instead of finding the optimal step size :math:`\alpha_{i}`, we can directly minimize
:math:`\rho` in the space spanned by :math:`\{x_{i}, r_{i}\}` via the Rayleigh-Ritz method and
identify the minimizing vector as :math:`x_{i+1}`. Furthermore, it is known that convergence of the
algorithm is drastically improved if we search :math:`x_{i+1}` in the extended space spanned by
:math:`\{x_{i}, x_{i-1}, r_{i}\}`. Thus, one iteration of gradient descent is given by the following
steps, with orthogonal :math:`\{x_{i}, y_{i}, r_{i}\}` (:math:`x_{i}, y_{i}` normal) as the
carryover from the previous iteration and :math:`R_A` indicating the Rayleigh-Ritz routine over
matrix :math:`A`:

.. math::

    p & \leftarrow \frac{r_{i}}{|r_{i}|}
    \theta, \kappa & \leftarrow R_A[x_{i}, y_{i}, p] \\
    s & \leftarrow \kappa_1 y_{i} + \kappa_2 p \\
    t & \leftarrow \frac{\kappa_0}{|s|} s - |s| x_{i} \\
    u & \leftarrow \kappa_0 x_{i} + s
    x_{i+1} & \leftarrow \frac{u}{|u|}
    y_{i+1} & \leftarrow \frac{t}{|t|}
    r_{i+1} & \leftarrow A x_{i+1} - \theta x_{i+1}.

The normal vector :math:`y_{i}` is orthogonal to :math:`x_{i}` and :math:`r_{i}` and lies in the
space spanned by :math:`\{x_{i}, x_{i-1}, r_{i}\}`.

Single-vector optimization
==========================

The B of LOBPCG refers to the algorithm's ability to determine multiple eigenvectors simultaneously
as a block. We have however chosen to compute just the ground state vector in this implementation,
eyeing running on extremely large vectors (memory requirement of LOBPCG scales with the number of
eigenvectors to compute). This choice opens up further memory-footprint optimizations in the
Rayleigh-Ritz subroutine.

In the Rayleigh-Ritz subroutine, we form the matrix

.. math::

    R_{jk} = w_{j}^{\dagger} A w_{k},

where :math:`w = {x, y, p}`, and diagonalize it. With an undetermined number of simultaneous vectors
in :math:`w`, we'd have to concatenate :math:`x`, :math:`y`, and :math:`p` (thus creating their
copies) and then numerically invert :math:`R`. Since we know that there are only three vectors, we
can construct :math:`R_{jk}` "by hand" and analytically invert the 3x3 matrix.

Distributed arrays
==================

This function works transparently over distributed (sharded) input :math:`v_0` if the callable
passed as the ``mat`` argument preserves the sharding in the output.

References
==========

[1]: https://en.wikipedia.org/wiki/LOBPCG

Single-vector LOBPCG API
========================

.. autosummary::
   :toctree: ../generated

   ground_locg
   eigenpair_2x2
   eigenpair_3x3
"""
from collections.abc import Callable
import logging
from typing import Optional
from numpy.typing import DTypeLike, NDArray
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh


def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = (),
    maxiter: int = 100,
    tol: Optional[float] = None,
    vspace: tuple[int, DTypeLike] | None = None,
    log_level: int = logging.WARNING
) -> tuple[float, NDArray, int]:
    r"""Single-vector LOBPCG.

    Args:
        mat: Matrix :math:`A`, either as an Array or a function :math:`x \mapsto Ax`.
        xinit: Initial vector. If given as an integer (requires ``vspace`` if ``mat`` is callable),
            a one-hot vector is created internally.
        args: Additional arguments to callable ``mat``.
        maxiter: Maximum number of gradient descent iterations.
        tol: Convergence condition.
        vspace: Specification (dimension, dtype) of the vector space. Required only when ``mat`` is
            a callable and ``vinit`` is an integer.
        log_level: Verbosity level.

    Returns:
        The smallest eigenvalue, its eigenvector, and the number of gradient descent iterations
        required to achieve the solution.
    """
    if callable(mat):
        return _ground_locg_callable(mat, xinit, args, maxiter, tol, vspace=vspace,
                                     log_level=log_level)
    return _ground_locg_matrix(mat, xinit, maxiter, tol, log_level=log_level)


@jax.jit(static_argnames=['maxiter', 'debug', 'log_level'])
def _ground_locg_matrix(
    mat: jax.Array,
    xinit: jax.Array,
    maxiter: int,
    tol: jax.Array | float | None,
    debug: bool = False,
    log_level: int = logging.WARNING
):
    vspace = None
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        vspace = (mat.shape[1], mat.dtype)
    return _ground_locg_callable(lambda x, a: _mm(a, x), xinit, (mat,), maxiter, tol,
                                 vspace=vspace, debug=debug, log_level=log_level)


@jax.jit(
    static_argnames=[
        'matvec',
        'maxiter',
        'vspace',
        'debug',
        'log_level'
    ]
)
def _ground_locg_callable(
    matvec: Callable[[jax.Array], jax.Array],
    xinit: jax.Array | int,
    args: tuple,
    maxiter: int,
    tol: jax.Array | float | None,
    vspace: tuple[int, DTypeLike] | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING
):
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        sharding = None
        if not (mesh := get_abstract_mesh()).empty:
            sharding = PartitionSpec(mesh.axis_names)
        xinit = (jax.lax.broadcasted_iota(xinit.dtype, (vspace[0],), 0, out_sharding=sharding)
                 == xinit).astype(vspace[1])

    if tol is None:
        tol = float(jnp.finfo(xinit.dtype).eps)

    def diagnostics(xcurr, ycurr, rcurr, theta, kappa=None, reltol=None, converged=None):
        vectors = jnp.stack((xcurr, ycurr, rcurr), axis=1)
        sas = _mm(vectors.conjugate().T, matvec(vectors, *args))
        axcurr = matvec(xcurr, *args)
        rho = jnp.dot(xcurr.conjugate(), axcurr, out_sharding=jax.typeof(theta).sharding).real

        if kappa is None:
            kappa = jnp.zeros(3, dtype=xcurr.dtype)
        if reltol is None:
            reltol = jnp.array(0.)
        if converged is None:
            converged = jnp.array(False)

        return {
            'x': xcurr,
            'y': ycurr,
            'r': rcurr,
            'theta': theta,
            'rho': rho,
            'kappa': kappa,
            'sas': sas,
            'reltol': reltol,
            'converged': converged
        }

    def rayleigh_ritz(*vectors):
        if get_abstract_mesh().empty:
            sharding = None
        else:
            sharding = PartitionSpec(None)

        nv = len(vectors)
        melems = jnp.zeros((nv, nv), dtype=vectors[0].dtype)
        mvs = []
        for iv1, v1 in enumerate(vectors):
            mv1 = matvec(v1, *args)
            mvs.append(mv1)
            for iv2 in range(iv1 + 1, nv):
                melems = melems.at[iv2, iv1].set(
                    jnp.dot(vectors[iv2].conjugate(), mv1, out_sharding=sharding)
                )
        melems += melems.conjugate().T
        for iv1, (v1, mv1) in enumerate(zip(vectors, mvs)):
            melems = melems.at[iv1, iv1].set(
                jnp.dot(v1.conjugate(), mv1, out_sharding=sharding)
            )

        if nv == 2:
            return eigenpair_2x2(melems)
        return eigenpair_3x3(melems)
        # A more streamlined (but memory-consuming) implementation:
        # vectors = jnp.stack(vectors, axis=1)
        # melems = _mm(vectors.conjugate().T, matvec(vectors, *args), out_sharding=sharding)
        # eigvals, eigvecs = jnp.linalg.eigh(SAS)
        # or
        # eigvecs, eigvals = jax.lax.linalg.eigh(melems, symmetrize_input=False)
        # return eigvals[0], eigvecs[:, 0]

    def body_iter1(xcurr, rcurr):
        norm_rcurr = jnp.linalg.norm(rcurr)
        tmp_p = rcurr / jnp.where(norm_rcurr == 0., 1., norm_rcurr)
        theta, kappa = rayleigh_ritz(xcurr, tmp_p)
        tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
        tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
        xnext = tmp_u / jnp.linalg.norm(tmp_u)
        ynext = tmp_t / jnp.linalg.norm(tmp_t)
        rnext = matvec(xnext, *args) - theta * xnext
        if debug:
            diag = diagnostics(xnext, ynext, rnext, theta, jnp.insert(kappa, 1, 0.))
            return xnext, ynext, rnext, diag
        return xnext, ynext, rnext

    def body(state):
        xcurr, ycurr, rcurr = state[-3:]
        if log_level <= logging.DEBUG:
            jax.debug.print('LOCG iteration {}', state[0])

        # Residual basis selection.
        # R is supposed to be already orthogonal to X, but we find that it's necessary to project
        # out with respect to both X and P to get good convergence of the residual.
        tmp_p = _project_out((xcurr, ycurr), rcurr)
        # Projected eigensolve.
        theta, kappa = rayleigh_ritz(xcurr, ycurr, tmp_p)
        # New vectors
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = jnp.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / norm_s) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xnext = tmp_u / jnp.linalg.norm(tmp_u)
        ynext = tmp_t / jnp.linalg.norm(tmp_t)
        axnext = matvec(xnext, *args)
        rnext = axnext - xnext * theta
        # Use the intermediate AX for relative tolerance.
        #
        # Comments from lobpcg_standard:
        # =========
        # I tried many variants of hard and soft locking [3]. All of them seemed
        # to worsen performance relative to no locking.
        #
        # Further, I found a more experimental convergence formula compared to what
        # is suggested in the literature, loosely based on floating-point
        # expectations.
        #
        # [2] discusses various strategies for this in Sec 5.3. The solution
        # they end up with, which estimates operator norm |A| via Gaussian
        # products, was too crude in practice (and overly-lax). The Gaussian
        # approximation seems like an estimate of the average eigenvalue.
        #
        # Instead, we test convergence via self-consistency of the eigenpair
        # i.e., the residual norm |r| should be small, relative to the floating
        # point error we'd expect from computing just the residuals given
        # candidate vectors.
        # =========
        reltol = jnp.linalg.norm(axnext) - theta
        reltol *= xcurr.shape[0]
        # Allow some margin for a few element-wise operations.
        reltol *= 10
        norm_rnext = jnp.linalg.norm(rnext)
        converged = norm_rnext < tol * reltol
        if log_level <= logging.DEBUG:
            jax.debug.print('Residual {}, reltol {}, converged: {}', norm_rnext, reltol, converged)

        state = (state[0] + 1, converged, theta, xnext, ynext, rnext)
        if debug:
            return state, diagnostics(xnext, ynext, rnext, theta, kappa, reltol, converged)
        return state

    if log_level <= logging.DEBUG:
        jax.debug.print('Performing first LOBPCG steps')

    norm_xinit = jnp.linalg.norm(xinit)
    xinit /= norm_xinit
    axinit = matvec(xinit, *args)
    rho = jnp.dot(xinit.conjugate(), axinit, out_sharding=jax.typeof(norm_xinit).sharding).real
    rinit = axinit - rho * xinit
    if debug:
        diag0 = diagnostics(xinit, jnp.zeros_like(xinit), rinit, rho)
        diag0 = jax.tree.map(lambda a: jnp.expand_dims(a, 0), diag0)

    vs_iter1 = body_iter1(xinit, rinit)
    if debug:
        diag1 = vs_iter1[-1]
        diag1 = jax.tree.map(lambda a: jnp.expand_dims(a, 0), diag1)

    state = (0, False, 0.) + vs_iter1[:3]
    if debug:
        state, diagnostics = jax.lax.scan(
            lambda s, _: body(s), state, length=maxiter
        )
        diagnostics = jax.tree.map(lambda d0, d1, dr: jnp.concatenate([d0, d1, dr], axis=0),
                                   diag0, diag1, diagnostics)
    else:
        state = jax.lax.while_loop(
            lambda s: jnp.logical_and(s[0] < maxiter, ~s[1]),
            body,
            state
        )

    niter = state[0]
    eigval = state[2]
    xfinal = state[3]
    if debug:
        return eigval, xfinal, niter, diagnostics
    return eigval, xfinal, niter


def _mm(a, b, precision=jax.lax.Precision.HIGHEST, out_sharding=None):
    if out_sharding is None:
        out_sharding = jax.typeof(b).sharding
    return jax.lax.dot(a, b, precision=(precision, precision), out_sharding=out_sharding)


def _project_out(basis, vector):
    if get_abstract_mesh().empty:
        sharding = None
    else:
        sharding = PartitionSpec(None)

    for _ in range(2):
        ips = []
        for vb in basis:
            ips.append(jnp.dot(vb.conjugate(), vector, out_sharding=sharding))
        for vb, ip in zip(basis, ips):
            vector -= vb * ip
        norm = jnp.linalg.norm(vector)
        vector /= jnp.where(norm == 0., 1., norm)

    # Comments from the original function:
    # ================
    # It's crucial to end on a subtraction of the original basis.
    # This seems to be a detail not present in [2], possibly because of
    # of reliance on soft locking.
    #
    # Near convergence, if the residuals R are 0 and our last
    # operation when projecting (X, P) out from R is the orthonormalization
    # done above, then due to catastrophic cancellation we may re-introduce
    # (X, P) subspace components into U, which can ruin the Rayleigh-Ritz
    # conditioning.
    #
    # We zero out any columns that are even remotely suspicious, so the invariant
    # that [basis, U] is zero-or-orthogonal is ensured.
    # ================
    for _ in range(2):
        ips = []
        for vb in basis:
            ips.append(jnp.dot(vb.conjugate(), vector, out_sharding=sharding))
        for vb, ip in zip(basis, ips):
            vector -= vb * ip

    # A more streamlined (but memory-consuming) implementation
    # basis = jnp.stack(basis, axis=1)
    # for _ in range(2):
    #     vector -= _mm(basis, _mm(basis.conjugate().T, vector, out_sharding=sharding))
    #     norm = jnp.linalg.norm(vector)
    #     vector /= jnp.where(norm == 0., 1., norm)
    # for _ in range(2):
    #     vector -= _mm(basis, _mm(basis.conjugate().T, vector, out_sharding=sharding))

    return vector * (jnp.linalg.norm(vector) >= 0.99).astype(vector.dtype)


@jax.jit
def eigenpair_2x2(mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the lowest eigenpair of a 2x2 matrix."""
    d = jnp.diagonal(mat).real
    det = jnp.prod(d) - jnp.square(jnp.abs(mat[1, 0]))
    tr = jnp.sum(d)
    eigval = (tr - jnp.sqrt(tr * tr - 4. * det)) * 0.5
    eigvec = jnp.array([(d[1] - eigval + mat[1, 0].conjugate()) / (d[0] - eigval + mat[1, 0]), -1.])
    # Somehow dividing by jnp.linalg.norm(eigvec) causes compile time to explode..
    # Apparently jnp.sum really hates small arrays
    eigvec /= jnp.sqrt(eigvec[0].real * eigvec[0].real + eigvec[0].imag * eigvec[0].imag + 1.)
    return eigval, eigvec


@jax.jit
def eigenpair_3x3(mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the lowest eigenpair of a 3x3 matrix computed via Cardano's method.

    Reference:
        J. Kopp, Efficient numerical diagonalization of hermitian 3 × 3 matrices,
        Int. J. Mod. Phys. C. 19, 523 (2008).
    """
    d = jnp.diagonal(mat).real
    modod = jnp.square(jnp.abs(mat[jnp.array([1, 2, 2]), jnp.array([0, 0, 1])]))
    c2 = -jnp.sum(d)
    c1 = jnp.sum(d * jnp.roll(d, 1)) - jnp.sum(modod)
    c0 = jnp.sum(d * modod[::-1])
    c0 -= jnp.prod(d)
    c0 -= 2. * (mat[0, 2] * mat[1, 0] * mat[2, 1]).real
    p = jnp.square(c2) - 3. * c1
    q = -13.5 * c0 - c2 * c2 * c2 + 4.5 * c2 * c1
    phi = jnp.atan2(
        jnp.sqrt(27. * (0.25 * jnp.square(c1) * (p - c1) + c0 * (q + 6.75 * c0))),
        q
    ) / 3.
    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)
    xmin = jnp.min(jnp.array([2. * cphi, -cphi - jnp.sqrt(3.) * sphi, -cphi + jnp.sqrt(3.) * sphi]))
    eigval = jnp.sqrt(p) / 3. * xmin - c2 / 3.
    v0 = mat[:, 1].at[1].subtract(eigval)
    v1 = mat[:, 2].at[2].subtract(eigval)
    eigvec = jnp.cross(v0, v1).conjugate()
    re = eigvec.real
    im = eigvec.imag
    # jnp.sum() hates small arrays
    eigvec /= jnp.sqrt(re[0] * re[0] + re[1] * re[1] + re[2] * re[2]
                       + im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
    return eigval, eigvec
