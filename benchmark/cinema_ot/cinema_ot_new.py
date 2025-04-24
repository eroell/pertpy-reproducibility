import time
import cProfile
import pertpy as pt
import scanpy as sc
import numpy as np
from scipy import sparse
import io
import pstats
rng = np.random.default_rng(42)

adata = pt.dt.cinemaot_example()
sc.pp.sample(adata, n=80_000, replace=True)
adata.X = adata.raw.X.copy()

# k_samples = 10
# mock_adata_collector = []
# for i in range(k_samples):
#     mock_adata = pt.dt.cinemaot_example()
#     jitter = rng.integers(0, 1000, size=mock_adata.raw.X.shape)
#     jitter[jitter > 80] = 0
#     jitter = np.log1p(jitter)
#     jitter = sparse.csr_matrix(jitter)
#     mock_adata.X = mock_adata.raw.X.copy()
#     mock_adata.X = mock_adata.X + jitter
#     mock_adata_collector.append(mock_adata)

# adata = sc.concat([adata] + mock_adata_collector)

sc.pp.pca(adata)

cot = pt.tl.Cinemaot()
# warm up cache & jit
# start = time.time()
# de = cot.causaleffect(
#     adata,
#     pert_key="perturbation",
#     control="No stimulation",
#     return_matching=True,
#     thres=1,
#     smoothness=3e-5,
#     eps=1e-3,
#     solver="Sinkhorn",
#     preweight_label="cell_type0528",
# )
# print(f"Time taken for warmup: {time.time() - start} seconds")

start = time.time()
profiler = cProfile.Profile()
profiler.enable()

de = cot.causaleffect(
    adata,
    pert_key="perturbation",
    control="No stimulation",
    return_matching=True,
    thres=1,
    smoothness=3e-5,
    eps=1e-3,
    solver="Sinkhorn",
    preweight_label="cell_type0528",
)

profiler.disable()
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
ps.dump_stats(filename="cinema_ot_new_2nd_call.prof")

runtime = time.time() - start
print(f"Runtime: {runtime:.2f} seconds")
