__all__ = ["xp"]
import os

USE_CPU = os.getenv("USE_CPU", "0") == "1"

if USE_CPU:
    import numpy as xp
else:
    import cupy as xp
