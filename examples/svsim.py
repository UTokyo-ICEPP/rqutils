"""Execute a state vector simulation on multiple GPUs or over MPI.

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
mpirun python svsim.py 32 --gpus mpi
----------------------------------------
"""
import os
import logging
from argparse import ArgumentParser
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from qiskit import QuantumCircuit, transpile
from rqutils.svsim import svsim

parser = ArgumentParser()
parser.add_argument('num_qubits', metavar='NUM', type=int, default=30, help='Number of qubits.')
parser.add_argument('--gpus', metavar='LIST',
                    help='Comma-separated list of device IDs (e.g. "0,1,2,3") or "mpi".')
parser.add_argument('--out', metavar='PATH', default='svsim_out.h5',
                    help='Output path. In the MPI mode, the output must be accessible from all '
                    'processes.')
options = parser.parse_args()

# Set the verbosity (JAX warning message about not finding TPU can be ignored)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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

# Construct the circuit
circuit = QuantumCircuit(options.num_qubits)
circuit.h(0)
for iq in range(options.num_qubits - 1):
    circuit.cx(iq, iq + 1)

circuit = transpile(circuit, initial_layout=list(range(options.num_qubits)),
                    basis_gates=['rx', 'ry', 'rz', 'rzz'])

# Run the simulation (state vector is sharded automatically)
final_state = svsim(circuit)

# How many nonzero elements are in the final state?
num_nonzero = jnp.sum(jnp.logical_not(jnp.isclose(final_state, 0.)))
logger.info('Number of nonzero elements in the GHZ state: %d', num_nonzero)

# Write the output
if options.gpus == 'mpi':
    # Writing array shards from multiple processes requires synchronization. Output path
    from mpi4py import MPI

    if (proc_id := jax.process_index()) == 0:
        # Head process creates the file and defines the dataset
        with h5py.File(options.out, 'w', libver='latest') as out:
            out.create_dataset('final_state', shape=final_state.shape, dtype=final_state.dtype)
    else:
        # Other processes wait for the greenlight
        MPI.COMM_WORLD.recv(source=proc_id - 1, tag=11)

    # Open the file with 'a' mode and write out the shards
    with h5py.File(options.out, 'a', libver='latest') as out:
        for shard in final_state.addressable_shards:
            logger.info('Writing shard %s from process %d', shard.index, proc_id)
            out['final_state'][shard.index] = shard.data

    # Send the greenlight to the next process
    if proc_id < jax.process_count() - 1:
        MPI.COMM_WORLD.send(1, dest=proc_id + 1, tag=11)

else:
    with h5py.File(options.out, 'w', libver='latest') as out:
        out.create_dataset('final_state', data=final_state)
