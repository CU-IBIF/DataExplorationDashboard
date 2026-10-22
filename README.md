# FeatureScope: data exploration dashboard for object × feature data

Interactive, pipeline-based exploration of high-dimensional per-object measurements:
segmentation outputs from multiplexed imaging (IMC, CODEX/PhenoCycler, CyCIF, MIBI),
cytometry, organelle and particle measurements, and post-processed spatial or
single-cell transcriptomics. Rows are objects (cells, nuclei, organelles…) and columns
are features.

* **Import** CSV/TSV/Parquet/Feather/Excel/`.h5ad`, with automatic detection of
  features, metadata, centroid (x, y) columns, batch/sample columns and IDs. Detected
  roles can be edited before import. Server-side paths are supported for large files on
  scratch.
* **Preprocessing:** NaN filtering and QC thresholds, gate-based subsetting, feature
  selection, imputation, arcsinh/log/Yeo-Johnson transforms, percentile clipping, z-score,
  robust, min-max, quantile, rank-Gauss and library-size normalization (per batch if
  wanted), highly variable features, covariate regression.
* **Batch correction:** per-batch centering/scaling (optionally to a reference batch),
  ComBat, Harmony, a batch-conditional VAE, and scVI (optional).
* **Dimensionality reduction:** PCA, UMAP, t-SNE, diffusion maps with pseudotime. These
  run on the GPU when rapids-singlecell is installed.
* **Clustering:** Leiden, Louvain, k-means, Gaussian mixture, FlowSOM-style SOM plus
  metaclustering, and marker features per cluster.
* **Deep-learning embeddings (PyTorch, GPU automatic):** a batch-aware variational
  autoencoder, a contrastive self-supervised encoder, a spatial graph neural network
  trained with Deep Graph Infomax (microenvironment/tissue-domain embedding), and scVI.
* **Graph and neighborhood analysis:** kNN and spatial graphs (kNN/radius/Delaunay,
  never crossing slides), neighborhood enrichment z-scores, cellular neighborhoods
  (Schürch et al.), multi-hop neighbor aggregation, PAGA-style cluster connectivity,
  and Moran's I.
* **Linked visualization and gating:** scatter/embedding, spatial, histogram, box by
  group, group × feature heatmap, and selection-vs-rest effect sizes. A lasso or box in
  any view highlights the same objects in every view. Gates can be polygon, rectangle,
  range or boolean, and are hierarchical (gate *within* a parent gate, as in FlowJo).
  Gates are saved with the project and can drive pipeline steps (`subset_gate`). Objects
  can also be labelled manually.
* **Save and reload without recomputation:** pipelines are JSON/YAML files and every
  step's output is cached under a content hash. Reopening a project, loading a saved
  pipeline, or fetching results computed on a cluster restores the results instantly.
  Editing step *k* recomputes only steps *k…n*.
* **Offload to HPC/cloud:** Slurm over SSH (Creighton HPC, UNL HCC Swan, ACCESS sites),
  Kubernetes (NRP Nautilus), AWS Batch and Google Cloud Batch, plus Open OnDemand and
  Apptainer templates. See **[docs/COMPUTE_STRATEGY.md](docs/COMPUTE_STRATEGY.md)**,
  which includes the options available to Creighton faculty and staff.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[deep,batch]"          # torch + harmonypy; drop [deep] for a lighter install
featurescope serve --workspace projects  # http://127.0.0.1:8765
```

In the browser, go to **Data → Create demo dataset**, then **Pipeline → Run pipeline**.
Lasso a population on any plot and watch it highlight everywhere. Use **Save gate** to
keep it.

Headless or on a cluster:

```bash
featurescope create myproj cells.parquet --batch slide          # or: featurescope create myproj demo
featurescope run myproj --pipeline-file examples/imaging.yaml --export
featurescope plan myproj --pipeline imaging                      # which steps are cached
featurescope submit myproj --pipeline imaging --profile hcc-swan --dry-run
```

Python API (results are plain AnnData, compatible with scanpy/squidpy):

```python
from featurescope import Project, Pipeline
p = Project.create("myproj", "cells.csv")
pl = Pipeline("std").add("transform", method="arcsinh", cofactor=150).add("normalize") \
                    .add("pca", n_comps=20).add("neighbors").add("umap").add("leiden")
adata, report = p.run(pl)        # second call: every step 'cached'
```

## Architecture

```
featurescope/
  data.py            import, role detection, column specs (feat:/raw:/obs:/obsm:)
  pipeline/          Step/Pipeline spec, op registry, Delta (per-step output), cached engine
  steps/             preprocess, batch, reduce, cluster, graph, deep  (all registered ops)
  gating.py          hierarchical polygon/rect/range/boolean gates
  project.py         on-disk project: data/base.h5ad, pipelines/, gates.json, cache/, results/, jobs/
  compute/           executors (slurm, kubernetes, aws_batch, gcp_batch) + job templates
  server/app.py      FastAPI: plotting payloads, linked selection, gates, pipelines, jobs
  web/               single-page UI (vanilla JS + vendored Plotly; works offline on HPC)
deploy/              Dockerfiles, Apptainer, Open OnDemand app, Slurm serve script, compute.example.yaml
examples/            ready-made pipelines (imaging, spatial niches, transcriptomics)
```

Every operation is a pure function `fn(adata, ctx, **params) -> Delta`. Adding a method
is one decorated function in `steps/`; it then appears in the UI's step list with a form
generated from its parameter schema.

Key design choices for large data:

1. The browser draws a stable display sample of at most 150k points, while
   selections, gates and statistics are evaluated server-side on all objects.
2. Deltas, not snapshots, are cached.
3. GPU backends are selected automatically.
4. Projects are self-contained directories that move between laptop, cluster and
   cloud.

## Tests

```bash
pip install -e ".[dev,deep,batch]" && pytest
```

The tests cover import and role detection, delta round-trips, cache hits and partial
invalidation, gates, an end-to-end analysis (checking that clusters and neighborhoods
recover the synthetic ground truth), the deep models, the HTTP API with linked
selection, dry-run job generation for every compute backend, and simulated
remote-then-fetch reuse of cached results.
