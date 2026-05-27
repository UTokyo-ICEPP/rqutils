"""Types and type hints."""
from collections.abc import Sequence
import numpy as np

Integer = int | np.integer
MatrixDimension = Integer | Sequence[Integer]
