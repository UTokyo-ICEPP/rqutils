r"""
========================================================================================
GPU-efficient symplectic representation of Pauli sums (:mod:`rqutils.paulis.symplectic`)
========================================================================================

.. currentmodule:: rqutils.paulis.symplectic

Symplectic representation of a Pauli string
===========================================

Any Pauli string :math:`Q` can be expressed as

.. math::

    Q = (-i)^{xz} \left(Z^{z_{n-1}} \otimes \cdots Z^{z_{0}}\right)
                                  \left(X^{x_{n-1}} \otimes \cdots X^{x_{0}}\right)

where :math:`x` (X signature) and :math:`z` (Z signature) are binary vectors of length :math:`n`
(number of qubits) and :math:`xz` represents their inner product.

Symplectic Pauli sum representation API
=======================================

.. autoclass:: PauliSumXZ
"""
from dataclasses import dataclass, field
from typing import Any
import warnings
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax.sharding import PartitionSpec, get_abstract_mesh
try:
    from qiskit.quantum_info import SparsePauliOp
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@register_dataclass
@dataclass(frozen=True)
class PauliSumXZ:
    """Symplectic (XZ) representation of a sum of Pauli strings."""
    x: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
    z: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
    c: np.ndarray[tuple[int, int], np.dtype[np.inexact]]
    num_qubits: int = field(metadata={'static': True})

    @classmethod
    def from_paulisum(
        cls,
        paulisum: Any,
        force_real: bool = False,
        add_padding: bool = False
    ) -> 'PauliSumXZ':
        if isinstance(paulisum, tuple):  # ([paulis], [coeffs])
            paulis, coeffs = paulisum
            if len(paulis) != len(coeffs):
                raise ValueError('Lengths of Pauli and coeff lists do not match')
            if len(set(len(p) for p in paulis)) != 1:
                raise ValueError('Pauli strings have non-uniform lengths')

            coeffs = np.array(coeffs)
            # Sort and consolidate the pauli strings and coefficients
            paulis = np.array([list(p.upper()) for p in paulis])
            nonzero = np.nonzero(coeffs)
            coeffs = coeffs[nonzero]
            paulis = paulis[nonzero]
            paulis, indices = np.unique(paulis, axis=0, return_inverse=True)
            masks = (indices[None, :] == np.arange(paulis.shape[0])[:, None]).astype(int)
            coeffs = masks @ coeffs
            xbits = np.logical_or(paulis == 'X', paulis == 'Y')
            zbits = np.logical_or(paulis == 'Y', paulis == 'Z')
            num_qubits = paulis.shape[1]

        elif HAS_QISKIT and isinstance(paulisum, SparsePauliOp):
            if not np.allclose(paulisum.coeffs.imag, 0.):
                raise ValueError('Coefficients of Paulis must be real for the Hamiltonian to be'
                                ' Hermitian.')

            # Remove null terms
            paulisum = paulisum.simplify()
            coeffs = paulisum.coeffs
            xbits = paulisum.paulis.x[:, ::-1]
            zbits = paulisum.paulis.z[:, ::-1]
            num_qubits = paulisum.num_qubits

        else:
            raise ValueError('Unsupported input type')

        if force_real:
            if np.any(coeffs.imag != 0.):
                warnings.warn('Found nonzero imaginary part when force_real=True')
            coeffs = coeffs.real

        # Find unique X signatures together with correspondence pointers
        xuniq, indices, counts = np.unique(xbits, axis=0, return_inverse=True, return_counts=True)
        xsignatures = xuniq.astype(np.uint8)
        # Group the Z signatures and coeffs by X signatures
        shape = (xsignatures.shape[0], np.max(counts))
        zsignatures = np.zeros(shape + zbits.shape[-1:], dtype=np.uint8)
        phcoeffs = np.zeros(shape, dtype=np.complex128)
        for isig, xsig in enumerate(xsignatures):
            ipaulis = np.nonzero(indices == isig)[0]
            zsigs = zbits[ipaulis].astype(np.uint8)
            zsignatures[isig, :counts[isig]] = zsigs
            # Multiply the coeffs by (-i)^{n_zx}
            iphases = np.sum(xsig & zsigs, axis=1) & 3
            phases = np.array([1., -1.j, -1., 1.j])[iphases]
            phcoeffs[isig, :counts[isig]] = coeffs[ipaulis] * phases

        if np.all(phcoeffs.imag == 0.):
            phcoeffs = phcoeffs.real

        if add_padding:
            # Add a dummy identity Pauli at the padding bit to align with the padding on the states
            xsignatures = np.pad(xsignatures, {1: (1, 0)})
            zsignatures = np.pad(zsignatures, {2: (1, 0)})

        # Pack the bit signatures
        xsignatures = np.packbits(xsignatures, axis=-1)
        zsignatures = np.packbits(zsignatures, axis=-1)
        return cls(xsignatures, zsignatures, phcoeffs, num_qubits)

    @property
    def arrays(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.x, self.z, self.c

    def matmul(self, rhs: NDArray):
        if rhs.shape[0] != 2 ** self.num_qubits:
            raise ValueError(f'RHS axis 0 size {rhs.shape[0]} is incompatible with'
                             f' num_qubits={self.num_qubits}')

        indices = jnp.arange(rhs.shape[0], dtype=np.int32, out_sharding=jax.typeof(rhs).sharding)
        powers = 256 ** jnp.arange(self.x.shape[1])[::-1]
        offset = 8 * self.x.shape[1] - self.num_qubits
        packed_x = jnp.sum(self.x * powers, axis=1, dtype=np.int32) >> offset
        packed_z = jnp.sum(self.z * powers, axis=2, dtype=np.int32) >> offset

        def apply_xgrp(out, data):
            xsig, zsigs, coeffs = data
            signs = 1. - 2. * (jnp.bitwise_count(indices & zsigs[:, None]) & 1)
            diags = jnp.sum(coeffs[..., None] * signs, axis=0)
            diags = jnp.expand_dims(diags, tuple(np.arange(1, rhs.ndim) + 1))
            out += rhs.at[indices ^ xsig].get(out_sharding=jax.typeof(rhs).sharding) * diags
            return out, None

        return jax.lax.scan(apply_xgrp, jnp.zeros_like(rhs), (packed_x, packed_z, self.c))[0]
