import numpy as np

from scalene import scalene_profiler

rng = np.random.default_rng(42)

a = rng.random((1000, 10000))

scalene_profiler.start()
b = rng.random((10000, 1000))

c = a @ b
scalene_profiler.stop()
d = c.sum(axis=1)
