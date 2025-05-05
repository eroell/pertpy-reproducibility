import cProfile
import io
import pstats
import time

import argparse
import muon as mu
import numpy as np
import pertpy as pt
import scanpy as sc
import pandas as pd


import muon as mu
import numpy as np
import pertpy as pt
import scanpy as sc
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--n_cells", type=int, default=2000)
parser.add_argument("--commit", type=str, default="na")
parser.add_argument("--rep", type=int, default=1)

args = parser.parse_args()

n_cells = args.n_cells
commit = args.commit
rep = args.rep

# from scalene import scalene_profiler
print(f"Sampling {n_cells} cells, rep {rep}, commit {commit}")

# Load dataset
mdata = pt.dt.papalexi_2021()
rng = np.random.default_rng(42)
choice = rng.choice(mdata["rna"].n_obs, size=n_cells, replace=True)

mdata = mdata[choice, :].copy()
mdata["rna"].obs_names_make_unique()
mdata["adt"].obs.index = mdata["rna"].obs.index.astype(str)
mdata["hto"].obs.index = mdata["rna"].obs.index.astype(str)
mdata["gdo"].obs.index = mdata["rna"].obs.index.astype(str)
# mdata.write_h5mu("papalexi_2021_1000.h5mu")
# mdata = mu.read_h5mu("/Users/eljas.roellin/Documents/pertpy_workspace/pertpy-reproducibility/benchmark/mixscape/papalexi_2021_1000.h5mu")

# JIT warmup of PyNNDescent
# mdata_mini = mdata[np.random.default_rng(42).choice(mdata["rna"].n_obs, size=1000, replace=False)].copy()
# mixscape_identifier = pt.tl.Mixscape()
# mixscape_identifier.perturbation_signature(
#     mdata_mini["rna"], "perturbation", "NT", split_by="replicate", n_neighbors=20, n_dims=40,
# )

# Start time
start_time = time.time()

# scalene_profiler.start()

# Preprocessing
# RNA
sc.pp.highly_variable_genes(mdata["rna"], n_top_genes=2000, flavor='seurat_v3', subset=True)
sc.pp.normalize_total(mdata["rna"])
sc.pp.log1p(mdata["rna"])
mdata["rna"].layers["scaled"] = mdata["rna"].X.copy()
sc.pp.scale(mdata["rna"], layer="scaled")

# Protein
mu.prot.pp.clr(mdata["adt"])

# Gene expression-based cell clustering UMAP
sc.pp.pca(mdata["rna"], n_comps=50, layer="scaled")
sc.pp.neighbors(mdata["rna"], metric="cosine")
sc.tl.umap(mdata["rna"])

# Mitigating confounding effects
mixscape_identifier = pt.tl.Mixscape()
mdata["rna"].X = mdata["rna"].X.toarray()

profiler = cProfile.Profile()
profiler.enable()

perturbation_start = time.time()
mixscape_identifier.perturbation_signature(
    mdata["rna"], "perturbation", "NT", split_by="replicate", n_neighbors=20, n_dims=40,
)
perturbation_end = time.time()
print(f"Perturbation signature took {perturbation_end - perturbation_start} seconds")


mixscape_start = time.time()
# Identify cells with no detectable perturbation
mixscape_identifier.mixscape(
    adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert"
)
mixscape_end = time.time()
print(f"Mixscape took {mixscape_end - mixscape_start} seconds")

# Visualizing perturbation responses with Linear Discriminant Analysis (LDA)
lda_start = time.time()
mixscape_identifier.lda(
    adata=mdata["rna"], control="NT", labels="gene_target"
)

lda_end = time.time()
print(f"LDA took {lda_end - lda_start} seconds")
profiler.disable()
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
ps.dump_stats(filename=f"20250505_mixscape_cluster_{n_cells}_{rep}_{commit}.prof")

# Compute and print elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")



