import os
import string
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from rqutils.ground_locg import ground_locg


@jax.jit
def matvec(vec, hx):
    num_qubits = np.round(np.log2(vec.shape[0])).astype(int)
    sharding = jax.typeof(vec).sharding

    def indices():
        return jax.lax.broadcasted_iota(np.int32, vec.shape, 0, out_sharding=sharding)

    def xvec(carry, bit):
        vec, result = carry
        source = indices() ^ (1 << bit)
        return (vec, result + vec.at[source].get()), None

    def zzvec(carry, bits):
        vec, result = carry
        mask = (1 << bits[0]) | (1 << bits[1])
        signs = (jnp.bitwise_count(indices() & mask) & 1) * 2. - 1.
        return (vec, result + vec * signs), None

    result = jnp.zeros_like(vec)
    result = jax.lax.scan(xvec, (vec, result), jnp.arange(num_qubits))[0][1]
    result *= hx
    bits = jnp.stack([jnp.arange(num_qubits), jnp.roll(jnp.arange(num_qubits), -1)], axis=1)
    result = jax.lax.scan(zzvec, (vec, result), bits)[0][1]
    return result


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

    vspace = (2 ** options.num_qubits, np.float64)

    hx_args = options.hx.split(',')
    hxs = np.linspace(float(hx_args[0]), float(hx_args[1]), int(hx_args[2]))

    def get_ground_states(_, hx):
        val0, vec0 = ground_locg(matvec, 0, args=(hx,), vspace=vspace, tol=1.e-14)[:2]
        val1 = ground_locg(matvec, vspace[0] - 1, args=(hx,), orth=(vec0,), vspace=vspace,
                           tol=1.e-12)[0]
        return None, jnp.stack([val0, val1])

    eigvals = jax.lax.scan(get_ground_states, None, hxs)[1]

    with h5py.File(options.out, 'w') as f:
        f.create_dataset('hxs', data=hxs)
        f.create_dataset('eigvals', data=eigvals)
