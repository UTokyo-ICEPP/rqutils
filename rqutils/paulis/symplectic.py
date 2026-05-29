"""
========================================================================================
GPU-efficient symplectic representation of Pauli sums (:mod:`rqutils.paulis.symplectic`)
========================================================================================

.. currentmodule: rqutils.paulis.symplectic

Symplectic Pauli sum representation API
=======================================

.. autosummary::
   :toctree: ../generated

   PauliSumXZ
   to_xzrep
"""
from dataclasses import dataclass
from typing import Any
import warnings
import numpy as np
from jax.tree_util import register_dataclass
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

    @classmethod
    def from_paulisum(
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
            xbits = (paulis == 'X' | paulis == 'Y')
            zbits = (paulis == 'Y' | paulis == 'Z')

        elif HAS_QISKIT and isinstance(paulisum, SparsePauliOp):
            if not np.allclose(hamiltonian.coeffs.imag, 0.):
                raise ValueError('Coefficients of Paulis must be real for the Hamiltonian to be'
                                ' Hermitian.')

            # Remove null terms
            hamiltonian = hamiltonian.simplify()
            coeffs = hamiltonian.coeffs
            xbits = hamiltonian.paulis.x[:, ::-1]
            zbits = hamiltonian.paulis.z[:, ::-1]

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
        return PauliSumXZ(xsignatures, zsignatures, phcoeffs)
