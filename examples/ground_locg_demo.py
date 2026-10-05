import os
import string
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec, NamedSharding
from rqutils.ground_locg import ground_locg


def get_shape_and_shardings(vec, qubit_partitioning):
    shape = tuple(2 ** np.array(qubit_partitioning))

    sharding = jax.typeof(vec).sharding
    if sharding.num_devices == 0:
        return shape, None, None

    in_spec = sharding.spec[0]  # ('X', 'Y', 'Z', ...)
    partitions = ()
    ipart = 0
    for nq in qubit_partitioning:
        partitions += (in_spec[ipart:ipart + nq],)
        ipart += nq
        if ipart >= len(in_spec):
            break
    return shape, sharding, NamedSharding(sharding.mesh, PartitionSpec(*partitions))


def make_matvec(num_qubits: int, axis_type: AxisType = AxisType.Auto):
    """Return a function that applies the 1D periodic TFIM Hamiltonian to a vector."""
    def make_apply_zz(qubit1, qubit2):
        def apply_zz(vec):
            qpart = (num_qubits - qubit2 - 1, 1, qubit2 - qubit1 - 1, 1, qubit1)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec *= jnp.array([[-1., 1.], [1., -1.]]).reshape((1, 2, 1, 2, 1))
            vec = jnp.reshape(vec, (2 ** num_qubits,), out_sharding=out_sharding)
            return vec

        return apply_zz

    def make_apply_x(qubit):
        def apply_x(vec):
            qpart = (num_qubits - qubit - 1, 1, qubit)
            if axis_type == AxisType.Explicit:
                shape, out_sharding, tmp_sharding = get_shape_and_shardings(vec, qpart)
            else:
                shape = tuple(2 ** np.array(qpart))
                out_sharding, tmp_sharding = None, None

            vec = jnp.reshape(vec, shape, out_sharding=tmp_sharding)
            vec = jnp.flip(vec, axis=1)
            vec = jnp.reshape(vec, (2 ** num_qubits,), out_sharding=out_sharding)
            return vec

        return apply_x

    zz_fns = [make_apply_zz(q1, q2) for q1, q2 in zip(range(num_qubits - 1), range(1, num_qubits))]
    zz_fns += [make_apply_zz(0, num_qubits - 1)]
    x_fns = [make_apply_x(q) for q in range(num_qubits)]

    @jax.jit
    def matvec(vec, hx):
        result = jnp.zeros_like(vec)
        for fn in x_fns:
            result += fn(vec)
        result *= hx
        for fn in zz_fns:
            result += fn(vec)
        return result

    return matvec


if __name__ == "__main__":
    import logging
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=30, help="Number of qubits")
    parser.add_argument("--hx", default='0.1,2.1,21', help="Transverse field strengths")
    parser.add_argument("--out", default='tfim.h5', help="Output file name")
    parser.add_argument("--gpus")
    parser.add_argument('--localmpi', action='store_true')
    options = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    if options.gpus:
        LOG.info('Parallelizing over %s', options.gpus)
        if options.gpus == 'mpi':
            from mpi4py import MPI
            jax.distributed.initialize(cluster_detection_method="mpi4py")
        elif options.localmpi:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            gpus = options.gpus.split(',')
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus[comm.Get_rank()]
            jax.distributed.initialize('localhost:10000', comm.Get_size(), comm.Get_rank())
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = options.gpus

        ngpu = jax.device_count()
        nax = np.log2(ngpu).astype(int)
        if 2 ** nax != ngpu:
            raise ValueError('Invalid ngpu')
        mesh_shape = (2,) * nax
        axis_names = tuple(string.ascii_lowercase[:nax])
        jax.set_mesh(jax.make_mesh(mesh_shape, axis_names, axis_types=(AxisType.Explicit,) * nax))

    matvec_fn = make_matvec(options.num_qubits, axis_type=AxisType.Explicit)
    vspace = (2 ** options.num_qubits, np.float64)

    hx_args = options.hx.split(',')
    hxs = np.linspace(float(hx_args[0]), float(hx_args[1]), int(hx_args[2]))

    eigvals, eigvecs = jax.lax.scan(
        lambda _, hx: (None, ground_locg(matvec_fn, 0, args=(hx,), maxiter=1000, vspace=vspace)[:2]),
        None,
        hxs
    )[1]

    with h5py.File(options.out, 'w') as f:
        f.create_dataset('hxs', data=hxs)
        f.create_dataset('eigvals', data=eigvals)
        f.create_dataset('eigvecs', data=eigvecs)
