"""Demonstration of ground_locg using 1D TFIM."""

import os
import string
import time
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec
from jax.experimental.mesh_utils import create_hybrid_device_mesh
from rqutils.ground_locg import ground_locg


@jax.jit
def matvec(vec: jax.Array, hx: jax.Array) -> jax.Array:
    """Linear operator that implements the Ising model Hamiltonian.
    
    Args:
        vec: State vector.
        hx: Transverse field value.
    Returns:
        Result of applying the Hamiltonian onto vec.
    """
    num_qubits = np.round(np.log2(vec.shape[0])).astype(int)
    sharding = jax.typeof(vec).sharding
    axis_names = sharding.spec[0]

    def reshaped_vec(partitioning):
        if sharding.num_devices == 0:
            return jnp.reshape(vec, tuple(2 ** nq for nq in partitioning))
        
        reshape_spec = ()
        nq = 0
        for psize in partitioning:
            reshape_spec += (axis_names[nq:nq + psize],)
            nq += psize
            if nq >= len(axis_names):
                break
        out_sharding = PartitionSpec(*reshape_spec)
        return jnp.reshape(vec, tuple(2 ** nq for nq in partitioning), out_sharding=out_sharding)

    result = jnp.zeros_like(vec)
    for qubit in range(num_qubits):
        term = reshaped_vec((num_qubits - qubit - 1, 1, qubit))
        term = jnp.flip(term, axis=1)
        result += jnp.reshape(term, (2 ** num_qubits,), out_sharding=sharding)

    result *= hx

    for qubit in range(num_qubits - 1):
        term = reshaped_vec((num_qubits - qubit - 2, 1, 1, qubit))
        term *= jnp.array([[-1., 1.], [1., -1.]]).reshape((1, 2, 2, 1))
        result += jnp.reshape(term, (2 ** num_qubits,), out_sharding=sharding)

    term = reshaped_vec((1, num_qubits - 2, 1))
    term *= jnp.array([[-1., 1.], [1., -1.]]).reshape((2, 1, 2))
    result += jnp.reshape(term, (2 ** num_qubits,), out_sharding=sharding)

    return result


if __name__ == "__main__":
    import logging
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=30, help="Number of qubits")
    parser.add_argument("--hx", default='0.1,2.1,21', help="Transverse field strengths")
    parser.add_argument("--max-level", type=int, default=1, help="Maximum level")
    parser.add_argument("--out", default='tfim.h5', help="Output file name")
    parser.add_argument("--gpus")
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
        elif options.localmpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            gpus = options.gpus.split(',')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus[comm.Get_rank()]
            jax.distributed.initialize('localhost:10000', comm.Get_size(), comm.Get_rank())
            loginfo('Local multi-process parallelization using GPUs %s', options.gpus)
        else:
            loginfo('Parallelizing over GPUs %s', options.gpus)
            os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus

        ndev = jax.device_count()
        nax = np.log2(ndev).astype(int)
        if 2 ** nax != ndev:
            raise ValueError('Invalid ngpu')
        axis_names = tuple(string.ascii_lowercase[:nax])

        if options.gpus == 'mpi':
            unique_slices = set(getattr(d, 'slice_index', 0) for d in jax.devices())
            num_slices = len(unique_slices)
            ndev_per_slice = ndev // num_slices
            nax_per_slice = np.log2(ndev_per_slice).astype(int)
            if 2 ** nax_per_slice != ndev_per_slice:
                raise ValueError('Invalid ndev_per_slice')

            inner_mesh_shape = (1,) * (nax - nax_per_slice) + (2,) * nax_per_slice
            outer_mesh_shape = (2,) * (nax - nax_per_slice) + (1,) * nax_per_slice
            loginfo('Inner mesh shape: %s, outer mesh shape: %s',
                    inner_mesh_shape, outer_mesh_shape)
            mesh = Mesh(create_hybrid_device_mesh(inner_mesh_shape, outer_mesh_shape),
                        axis_names=axis_names, axis_types=(AxisType.Explicit,) * nax)
        else:
            mesh = jax.make_mesh((2,) * nax, axis_names, axis_types=(AxisType.Explicit,) * nax)
        jax.set_mesh(mesh)

    vspace = (2 ** options.num_qubits, np.float64)

    if ',' in options.hx:
        hx_args = options.hx.split(',')
        hxs = np.linspace(float(hx_args[0]), float(hx_args[1]), int(hx_args[2]))
    else:
        hxs = np.array([float(options.hx)])

    if jax.process_index() == 0:
        with h5py.File(options.out, 'w') as out:
            out.create_dataset('hxs', data=hxs)
            out.create_dataset('eigvals', shape=hxs.shape + (options.max_level + 1,),
                               dtype=np.float64)

    @jax.jit
    def get_ground_states(hx):
        vals = []
        orth = ()
        init = 0
        tol = 1.e-14
        for _ in range(options.max_level):
            val, vec = ground_locg(matvec, init, args=(hx,), orth=orth, vspace=vspace,
                                   tol=tol)[:2]
            vals.append(val)
            orth += (vec,)
            init = vspace[0] - 1
            tol *= 10.
        vals.append(ground_locg(matvec, init, args=(hx,), orth=orth, vspace=vspace, tol=tol)[0])
        return jnp.stack(vals)

    for ihx, hx in enumerate(hxs):
        loginfo('Starting hx=%.2f', hx)
        start = time.time()
        eigvals = np.array(get_ground_states(hx))
        loginfo('Done in %.2f seconds. eigvals=%s', time.time() - start, eigvals)
        if jax.process_index() == 0:
            with h5py.File(options.out, 'a') as out:
                out['eigvals'][ihx] = eigvals
