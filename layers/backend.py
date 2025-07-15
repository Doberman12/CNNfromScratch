import os

if os.getenv("USE_CPU", "0") == "1":
    import numpy as xp
else:
    import cupy as xp
