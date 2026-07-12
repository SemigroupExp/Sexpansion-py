"""Shared array type aliases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
