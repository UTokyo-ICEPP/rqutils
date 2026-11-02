"""Sample-based quantum diagonalization example.

This script will generate a random Hamiltonian for a given number of qubits, and perform a subspace
diagonalization by projecting it over randomly chosen Hilbert space dimensions.

To run in an HPC using MPI, define a job that spans 2^n nodes and execute this script under mpirun
from each node. With PBS (e.g. Miyabi), the job script would look like this:

job.sh
----------------------------------------
#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=group
#PBS -j oe
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=01:00:00

source venv/bin/activate
cd rqutils/examples
mpirun python sqd.py 32 --gpus mpi
----------------------------------------
"""
import os
import logging
from argparse import ArgumentParser
import numpy as np
import jax
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from qiskit.quantum_info import SparsePauliOp
from rqutils.sqd import SQDOptions, run_sqd
from rqutils.paulis.symplectic import PauliSumXZ

parser = ArgumentParser()
parser.add_argument('num_qubits', metavar='NUM', type=int, default=30, help='Number of qubits.')
parser.add_argument('--tfim', metavar='N', type=int,
                    help='Diagonalize a N-qubit TFIM Hamiltonian embedded in the larger num_qubits '
                         'Hilbert space.')
parser.add_argument("--hx", type=float, default=1., help="Transverse field strength")
parser.add_argument('--num-paulis', metavar='NUM', type=int, default=100,
                    help='Number of terms in the Hamiltonian.')
parser.add_argument('--subspace-frac', metavar='FRACTION', type=float, default=0.01,
                    help='Relative dimensionality of the projection subspace.')
parser.add_argument('--x-cache', type=int, default=1, help='X cache strategy.')
parser.add_argument('--z-cache', type=int, default=0, help='Z cache strategy.')
parser.add_argument('--shard-bundle', type=int, default=0,
                    help='Number of shards to bundle when caching X sources.')
parser.add_argument('--maxiter', type=int, default=1000, help='Maxiter parameter of LOCG.')
parser.add_argument('--tol', type=float, help='LOCG convergence tolerance.')
parser.add_argument('--gpus', metavar='LIST',
                    help='Comma-separated list of device IDs (e.g. "0,1,2,3") or "mpi".')
parser.add_argument('--localmpi', action='store_true')
options = parser.parse_args()

jax.config.update('jax_enable_x64', True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOG = logging.getLogger(__name__)

def loginfo(msg, *args):
    if jax.process_index() == 0:
        LOG.info(msg, *args)

if options.gpus:
    if options.gpus == 'mpi':
        from mpi4py import MPI
        jax.distributed.initialize(cluster_detection_method="mpi4py")
        loginfo('Parallelizing over MPI')

        axis_names = ('x', 'y')
        unique_slices = set(getattr(d, 'slice_index', 0) for d in jax.devices())
        num_slices = len(unique_slices)
        ndev_per_slice = jax.device_count() // num_slices
        inner_mesh_shape = (1, ndev_per_slice)
        outer_mesh_shape = (num_slices, 1)
        loginfo('Inner mesh shape: %s, outer mesh shape: %s', inner_mesh_shape, outer_mesh_shape)
        mesh = Mesh(create_hybrid_device_mesh(inner_mesh_shape, outer_mesh_shape),
                    axis_names=axis_names, axis_types=(AxisType.Explicit,) * 2)

    else:
        if options.localmpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            gpus = options.gpus.split(',')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus[comm.Get_rank()]
            jax.distributed.initialize('localhost:10000', comm.Get_size(), comm.Get_rank())
            loginfo('Local multi-process parallelization using GPUs %s', options.gpus)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus
            loginfo('Parallelizing over GPUs %s', options.gpus)

        axis_names = ('x',)
        mesh = jax.make_mesh((jax.device_count(),), axis_names, axis_types=(AxisType.Explicit,))

    jax.set_mesh(mesh)

# Initialize a pseudorandom number generator
rng = np.random.default_rng()

if options.tfim:
    embedding = np.sort(rng.choice(options.num_qubits, options.tfim, replace=False))
    num_bytes = np.ceil((options.num_qubits + 1) / 8).astype(int)

    num_samples = 2 ** options.tfim
    states_p = np.zeros((num_samples, num_bytes), dtype=np.uint8)
    indices = np.arange(num_samples)
    for ibit, pos in enumerate(embedding):
        ibyte = num_bytes - pos // 8 - 1
        states_p[:, ibyte] |= (((indices >> ibit) & 1) << (pos % 8)).astype(np.uint8)

    paulis = []
    for pos1, pos2 in zip(embedding[:-1], embedding[1:]):
        paulis.append('I' * (options.num_qubits - pos2 - 1)
                      + 'Z'
                      + 'I' * (pos2 - pos1 - 1)
                      + 'Z'
                      + 'I' * pos1)
    paulis.append('I' * (options.num_qubits - embedding[-1] - 1)
                  + 'Z'
                  + 'I' * (embedding[-1] - embedding[0] - 1)
                  + 'Z'
                  + 'I' * embedding[0])
    for pos in embedding:
        paulis.append('I' * (options.num_qubits - pos - 1) + 'X' + 'I' * pos)
    
    coeffs = [-1.] * options.tfim + [options.hx] * options.tfim

else:
    # Projection subspace is specified as a binary array of type uint8.
    # Array ordering follows the Qiskit convention (big endian): the rightmost digit is the least
    # significant and corresponds to qubit 0.
    num_samples = int((2 ** options.num_qubits) * options.subspace_frac)
    states = rng.choice(2, size=(num_samples, options.num_qubits)).astype(np.uint8)
    states_p = np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)

    # Construct random Paulis and coefficients
    paulis = [''.join('IXYZ'[ip] for ip in row)
              for row in rng.choice(4, size=(options.num_paulis, options.num_qubits))]
    coeffs = rng.uniform(-1., 1., size=options.num_paulis)
    
hamiltonian = PauliSumXZ.from_paulisum(SparsePauliOp(paulis, coeffs))

# Run SQD
options = SQDOptions(
    return_eigvec=False,
    cache_strategy=(options.x_cache, options.z_cache),
    shard_bundle=options.shard_bundle,
    locg_maxiter=options.maxiter,
    locg_tol=options.tol
)
eigval, = run_sqd(hamiltonian, states_p, states_p.shape[0], options=options)
print(eigval)
