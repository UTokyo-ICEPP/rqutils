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
from jax.sharding import AxisType
from qiskit.quantum_info import SparsePauliOp
from rqutils.sqd import sqd

parser = ArgumentParser()
parser.add_argument('num_qubits', metavar='NUM', type=int, default=30, help='Number of qubits.')
parser.add_argument('--num-paulis', metavar='NUM', type=int, default=100,
                    help='Number of terms in the Hamiltonian.')
parser.add_argument('--subspace-frac', metavar='FRACTION', type=float, default=0.01,
                    help='Relative dimensionality of the projection subspace.')
parser.add_argument('--gpus', metavar='LIST',
                    help='Comma-separated list of device IDs (e.g. "0,1,2,3") or "mpi".')
options = parser.parse_args()

# Set the verbosity (JAX warning message about not finding TPU can be ignored)
logging.basicConfig(level=logging.INFO)

# Enable 64-bit compute
jax.config.update('jax_enable_x64', True)

if options.gpus == 'mpi':
    jax.distributed.initialize(cluster_detection_method='mpi4py')
elif options.gpus is not None:
    # Specify the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus

if jax.device_count() > 1:
    # Set the array layout over the processes / devices
    jax.set_mesh(jax.make_mesh((jax.device_count(),), ('x',), (AxisType.Explicit,)))

# Initialize a pseudorandom number generator
rng = np.random.default_rng()

# Projection subspace is specified as a binary array of type uint8.
# Array ordering follows the Qiskit convention (big endian): the rightmost digit is the least
# significant and corresponds to qubit 0.
num_samples = int((2 ** options.num_qubits) * options.subspace_frac)
states = rng.choice(2, size=(num_samples, options.num_qubits)).astype(np.uint8)

# Construct random Paulis and coefficients
paulis = [''.join('IXYZ'[ip] for ip in row)
          for row in rng.choice(4, size=(options.num_paulis, options.num_qubits))]
coeffs = rng.uniform(-1., 1., size=options.num_paulis)
hamiltonian = SparsePauliOp(paulis, coeffs)

# Run SQD
eigval, eigvec, subdims = sqd(hamiltonian, states, return_eigvec=True)
