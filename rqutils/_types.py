from typing import Union, Sequence
from numbers import Number

import numpy as np
ArrayType = np.ndarray
try:
    import jax.numpy as jnp
except ImportError:
    pass
else:
    ArrayType = Union[ArrayType, jnp.ndarray]

array_like = Union[Number, Sequence[Number], ArrayType]

MatrixDimension = Union[int, Sequence[int]]
