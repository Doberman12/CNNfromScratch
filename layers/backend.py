import os

USE_CPU = os.getenv("USE_CPU", "0") == "1"

if USE_CPU:
    import numpy as xp
    from numpy.lib.stride_tricks import as_strided  # noqa: F401
else:
    import cupy as xp
    from cupy.lib.stride_tricks import as_strided  # noqa: F401
