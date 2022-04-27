from typing import Union

import numpy as np
ndarray = np.ndarray
try:
    import jax.numpy as jnp
except ImportError:
    pass
else:
    ndarray = Union[ndarray, jnp.ndarray]

array_like = Union[list, tuple, ndarray]
