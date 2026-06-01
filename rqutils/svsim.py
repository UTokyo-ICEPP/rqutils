"""
===================================================
Simple statevector simulator (:mod:`rqutils.svsim`)
===================================================
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax.sharding import NamedSharding, PartitionSpec, get_abstract_mesh
try:
    from qiskit.circuit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@register_dataclass
@dataclass
class CircuitXZ:
    """Symplectic (XZ) representation of a series of rotation gates."""
    x: np.ndarray[tuple[int], np.dtype[np.int64]]
    z: np.ndarray[tuple[int], np.dtype[np.int64]]
    cos: np.ndarray[tuple[int], np.dtype[np.floating]]
    sin: np.ndarray[tuple[int], np.dtype[np.floating]]
    num_qubits: int = field(metadata={'static': True})

type GateSpec = tuple[str, int | Sequence[int]] | tuple[str, int | Sequence[int], Any]
type CircuitInput = CircuitXZ | list[GateSpec]


def svsim(
    circuit: CircuitInput,
    initial_state: NDArray[np.complex128] | int = 0,
    out_sharding: Optional[NamedSharding | PartitionSpec] = None
):
    if not isinstance(circuit, CircuitXZ):
        circuit = to_circuitxz(circuit)

    return do_svsim(circuit, initial_state, out_sharding)


@jax.jit(static_argnames=['out_sharding'])
def do_svsim(
    circuit: CircuitXZ,
    initial_state: NDArray[np.complex128] | int = 0,
    out_sharding: Optional[NamedSharding | PartitionSpec] = None
):
    if out_sharding is None and not (mesh := get_abstract_mesh()).empty:
        out_sharding = PartitionSpec(mesh.axis_names)

    indices = jnp.arange(2 ** circuit.num_qubits, dtype=np.int64, out_sharding=out_sharding)

    if len(initial_state.shape) == 0:
        initial_state = (indices == initial_state).astype(np.complex128)

    def apply_gate(state, gate):
        signs = 1. - 2. * (jnp.bitwise_count(indices & gate.z) & 1)
        xstate = jax.lax.cond(
            jnp.all(gate.x == 0),
            lambda: state,
            lambda: state.at[indices ^ gate.x].get(out_sharding=out_sharding)
        )
        out = 1.j * gate.sin * signs * xstate
        out = jax.lax.cond(
            gate.cos == 0.,
            lambda: out,
            lambda: out + gate.cos * state
        )
        return out, None

    return jax.lax.scan(
        apply_gate,
        initial_state,
        circuit
    )[0]


def to_circuitxz(circuit: CircuitInput):
    """Translate circuit data given as a list of GateSpecs or a QuantumCircuit into signatures."""
    num_qubits = None

    if HAS_QISKIT and isinstance(circuit, QuantumCircuit):
        def qidx(qubits):
            if isinstance(qubits, tuple):
                return np.array(list(map(circuit.qregs[0].index, qubits)))
            else:
                return np.array([circuit.qregs[0].index(qubits)])

        gate_specs = []
        for datum in circuit.data:
            if (op := datum.operation).name == 'cz':
                gate_specs.extend([
                    ('rzz', qidx(datum.qubits), np.pi / 2.),
                    ('rz', qidx(datum.qubits[0]), -np.pi / 2.),
                    ('rz', qidx(datum.qubits[1]), -np.pi / 2.)
                ])
            elif op.params:
                gate_specs.append((op.name, qidx(datum.qubits), op.params[0]))
            else:
                gate_specs.append((op.name, qidx(datum.qubits)))

        num_qubits = circuit.num_qubits
        circuit = gate_specs
        

    xarr = np.zeros(len(circuit), dtype=np.int64)
    zarr = np.zeros(len(circuit), dtype=np.int64)
    cosarr = np.zeros(len(circuit), dtype=np.float64)
    sinarr = np.zeros(len(circuit), dtype=np.float64)
    qmax = 0
    for igate, gate in enumerate(circuit):
        match gate[0]:
            case 'x':
                spec = (1, 0, 'pi')
            case 'y':
                spec = (1, 1, 'pi')
            case 'z':
                spec = (0, 1, 'pi')
            case 'rx':
                spec = (1, 0, gate[2])
            case 'ry':
                spec = (1, 1, gate[2])
            case 'rz':
                spec = (0, 1, gate[2])
            case 'rzz':
                spec = (0, 1, gate[2])
            case _:
                raise ValueError(f'Unsupported gate name {gate[0]}')

        qubits = np.asarray(gate[1])
        xarr[igate] = np.sum(np.array(spec[0], dtype=np.int64) << qubits)
        zarr[igate] = np.sum(np.array(spec[1], dtype=np.int64) << qubits)
        if spec[2] == 'pi':
            sinarr[igate] = -1.
        else:
            cosarr[igate] = np.cos(-spec[2] * 0.5)
            sinarr[igate] = np.sin(-spec[2] * 0.5)

        qmax = max(qmax, np.max(qubits) + 1)

    if num_qubits is None:
        num_qubits = qmax

    return CircuitXZ(xarr, zarr, cosarr, sinarr, num_qubits)
