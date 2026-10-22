# Compute strategy: local, Creighton/regional HPC, national platforms, cloud

FeatureScope is built so that **where** a pipeline runs does not change **what** you get.
A pipeline is a JSON/YAML list of steps; every finished step is stored in
`<project>/cache/<key>/`, where the key is a hash of the input data fingerprint plus
every upstream step and its parameters. So:

* Results computed on a cluster, copied back as `cache/` entries, load in the dashboard
  instantly. Nothing is recomputed, and nothing has to be "imported".
* An interrupted or timed-out job loses at most the step that was running. Resubmitting
  continues from the last finished step.
* Uploads are incremental. The remote project directory is named after the dataset
  fingerprint, so `base.h5ad` is transferred once and later runs move only new cache
  entries.

```
 laptop / workstation                       cluster / cloud
 ┌───────────────────────┐   stage (rsync/S3)  ┌──────────────────────────────┐
 │ dashboard (FastAPI +  │ ─────────────────▶  │ project copy (same layout)   │
 │ browser), project dir │                     │ featurescope run --device cuda│
 │   cache/<key>/ …      │ ◀─────────────────  │   → cache/<key>/ …           │
 └───────────────────────┘   fetch cache/      └──────────────────────────────┘
```

## 1. What actually needs offloading

Measured on this repo's synthetic imaging data (16 features, 10 slides) on a 4-core,
16 GB container (no GPU). See `BENCHMARKS` below for the 1,000,000-object run.

| Step class | Scaling | 1M objects, 4 CPU cores | Recommendation |
|---|---|---|---|
| Import, transform, normalize, batch centering, ComBat | O(n·p), in-memory | seconds | local |
| PCA, k-means (mini-batch), FlowSOM, spatial kNN, cellular neighborhoods | ~O(n·p·k) | seconds to a minute | local |
| kNN graph (pynndescent), UMAP, Leiden | O(n log n), memory-heavy | many minutes; RAM ≈ 6× the matrix | HPC CPU node (16–64 cores), or a GPU with rapids-singlecell (10–100× faster) |
| VAE / contrastive / spatial GNN (DGI), scVI | many epochs × n | slow on CPU | GPU node |
| Dashboard interaction (plots, lasso, gates) | display sample ≤150k points; selections over all n | <1 s per interaction at 1M | local, or run the whole dashboard on a node (Section 4) |

The **Compute** tab estimates matrix size and RAM, and suggests a target from the
pending steps (`featurescope.compute.recommend`).

Rules of thumb: below about 200k objects, run everything locally. Between 200k and a
few million objects, send the kNN/UMAP/deep steps out and do everything else locally.
At tens of millions of objects, run the whole dashboard on a large-memory node through
Open OnDemand or an SSH tunnel.

## 2. Execution backends implemented

| Profile `type` | Mechanism | Credentials used | Good for |
|---|---|---|---|
| `slurm` | `rsync` over SSH → `sbatch` (optionally `apptainer exec --nv`) → `squeue`/`sacct` → `rsync` back | your SSH key; Duo/MFA once per 8 h through SSH ControlMaster | Creighton HPC, UNL HCC Swan, ACCESS sites |
| `kubernetes` | stage to S3 (Ceph) → `kubectl apply` a `batch/v1` Job → container syncs `cache/` back | kubeconfig plus an S3 key stored as a k8s secret | NRP Nautilus (GP-ENGINE), institutional k8s |
| `aws_batch` | stage to S3 → `aws batch submit-job` | AWS SSO/CLI profile | burst GPU, grant-funded cloud |
| `gcp_batch` | stage to GCS → `gcloud batch jobs submit` | gcloud auth | same |

Each submission writes `jobs/<id>/` containing `pipeline.json`, the generated script or
manifest (`job.sbatch`, `job.yaml`, `gcp-batch.json`), and a `commands.log` of every
command run. **Dry-run** (the default in the UI) writes all of this without submitting,
so it can be reviewed or run by hand.

Profiles go in `~/.featurescope/compute.yaml` or `<project>/compute.yaml`. Start from
`deploy/compute.example.yaml`.

### Container images

* `deploy/Dockerfile` is the CPU image. `deploy/Dockerfile.gpu` adds CUDA PyTorch,
  RAPIDS and rapids-singlecell.
* `deploy/apptainer.def` converts the image to a `.sif` for clusters, where Docker is
  normally not allowed. Run it with `apptainer exec --nv`.
* The same image serves Slurm, Kubernetes and cloud batch, so results are reproducible
  across sites.

> The Dockerfiles were not built in the development environment, which had no Docker
> daemon. Build and smoke-test them (`featurescope run` on the demo project) before
> relying on them, and pick the RAPIDS/CUDA tag that matches the cluster's driver.

## 3. Institutional and regional options for Creighton faculty and staff

What follows is what could be verified from public sources in October 2026. Creighton's
own HPC web page could not be fetched from the build environment (DNS failure), and
none of the public pages list hardware, partitions or login hostnames. **Every hostname
in `compute.example.yaml` marked `VERIFY` is a placeholder until confirmed with the
provider.**

| Option | Who can use it | Cost | GPUs | What we know / what to ask |
|---|---|---|---|---|
| **Creighton DoIT High Performance Computing** | Creighton faculty, staff and students | (ask DoIT) | (ask DoIT) | DoIT states that it offers high-performance research computing to faculty, staff and students, covering large data storage, high-speed transfer and increased compute, with consulting from the **RaD Lab** (now under Emerging Technologies, partnered with University Libraries). **Ask DoIT/RaD Lab:** login host, scheduler (Slurm?), GPU partitions and models, scratch and project storage quotas, Apptainer availability, Open OnDemand, Globus endpoint, and whether data classified as PHI is permitted. Request through the DoIT service portal (myServices). |
| **UNL Holland Computing Center (HCC), Swan cluster** | NU campuses by default. A Great Plains Network page states that HCC access for other Nebraska institutions is by agreement between the NU Board of Regents and the institution, and names **Creighton** as one such institution. | Shared access is free for NU. Priority access is cost-recovery. Terms for Creighton: ask. | Yes. Swan is HCC's main GPU resource, and the newer PLUMAGE GPU resource is part of Swan. | Accounts belong to a faculty-led group. HCC's current pages list only NU campuses, so **email hcc-support@unl.edu** to confirm whether the Creighton agreement is active and how to request a group. Swan has an Open OnDemand portal and uses Slurm. The `hcc-swan` profile in the example config targets it. |
| **NRP Nautilus (National Research Platform)**, regionally through **GP-ENGINE** / Great Plains Network | Researchers at non-profit research and education institutions | Free | Large pool. HCC alone operates 41 nodes and 204 GPUs for NRP. | Kubernetes. Sign in at portal.nrp.ai, then join or create a namespace (a faculty member typically requests the namespace). It provides JupyterHub, S3 (Ceph) storage, and batch Jobs. GP-ENGINE and HCC run onboarding workshops ("Migrating AI/ML workflows to Nautilus"). Best fit for the **GPU deep-embedding steps**. Not for PHI. |
| **NSF ACCESS** | Any US-based researcher or educator, including Creighton faculty | Free allocations | Yes (e.g. Purdue Anvil, SDSC Expanse, TACC, Delta) | "Explore" allocations are fast to obtain and suit pilots. Larger allocations need a short proposal. Every site runs Slurm, so the `slurm` profile works unchanged. |
| **OSG OSPool** | US academic researchers | Free | Limited | High-throughput, many small jobs. Useful for parameter sweeps (e.g. Leiden resolutions or VAE hyper-parameters split into independent jobs), less so for one large-memory kNN. |
| **Commercial cloud** (AWS, GCP, Azure) | Anyone with a funded account | Pay per use. Vendor research-credit programs exist. | On demand | Use when you need burst capacity, a specific GPU, or a deadline. Confirm with Creighton DoIT/Research Compliance whether an institutional agreement (and a BAA, for PHI) exists before placing human-subject data in a cloud account. |

### Recommended path for a Creighton lab

1. **Today, with no accounts:** run locally. Up to a few hundred thousand objects, the
   whole pipeline including small deep models runs on a workstation.
2. **Week 1:** open a DoIT ticket for HPC access, and email HCC support about the
   Creighton/Swan agreement. In parallel, request an **ACCESS Explore** allocation (free,
   fast) as a guaranteed Slurm + GPU fallback.
3. **For GPU deep learning:** get an NRP Nautilus namespace (free) for the VAE,
   contrastive and spatial GNN steps on de-identified data.
4. **Data governance:** keep PHI and identifiable human-subject data on Creighton-approved
   storage and compute. Object-level feature tables are usually de-identified, but slide
   and sample identifiers can link back to patients, so check with the IRB/DoIT before
   moving them to NRP, OSPool or a personal cloud account.
5. **Interactive use of very large data:** if Creighton's or HCC's cluster runs Open
   OnDemand, install `deploy/ood/featurescope` as an OnDemand interactive app. The
   dashboard then runs on a compute node and is used from the browser with no
   tunneling. Otherwise use `deploy/slurm/serve_on_node.sbatch` with an SSH tunnel.

### Questions to send DoIT / RaD Lab (copy-paste)

* Login hostname, scheduler, and partition names. Are there GPU nodes, and which GPU
  models and how many?
* Default and maximum job wall time, plus memory per node (we need ≥128 GB for kNN/UMAP
  on ~5M objects).
* Scratch and project storage paths and quotas. Is there a Globus endpoint?
* Is Apptainer/Singularity available? Can users run containers from GHCR or Docker Hub?
* Is Open OnDemand deployed? Can labs add their own interactive apps?
* Is SSH key authentication allowed (with Duo), and are ControlMaster connections
  permitted?
* Which data classifications (PHI, de-identified human data) may be stored and processed?
* Is the Creighton–UNL HCC agreement active, and who at Creighton coordinates it?

## 4. Two ways to use remote compute

**A. Offload steps, keep the dashboard local.** Use the Compute tab → profile → Submit →
↻ to poll. When the job reports `COMPLETED` the `cache/` is fetched and the new
embeddings and clusters appear. From the CLI:

```bash
featurescope submit myproj --pipeline imaging --profile hcc-swan --wait
featurescope jobs myproj --refresh
```

**B. Run the dashboard next to the data.** Use this for tens of millions of objects, or
when data may not leave the cluster.

* Open OnDemand: install `deploy/ood/featurescope/` (the app uses relative URLs and
  `--root-path`, so it works behind `/rnode/<host>/<port>/`).
* Plain Slurm: `sbatch deploy/slurm/serve_on_node.sbatch`, then use the printed
  `ssh -L` command.

## 5. Scaling techniques built into the app

* **Display sampling with full-data semantics.** Plots draw a stable random sample (the
  same objects in every view), while lasso/box selections, gates, histograms, box plots
  and heatmaps are computed server-side on all objects. A selection travels to the
  browser as a packed bitmask (1M objects = 125 kB).
* **Binary transport.** Coordinates are sent as base64 float32 and rendered with WebGL
  (`scattergl`).
* **Per-step deltas, not snapshots.** A clustering step stores one column. An embedding
  stores n×k floats. Only steps that rewrite X store a matrix.
* **GPU backends.** PCA, kNN, UMAP, t-SNE and Leiden switch to RAPIDS/rapids-singlecell
  when available (`backend: auto`). PyTorch steps use CUDA or Apple MPS automatically.
* **Mini-batch algorithms** above 100k objects (mini-batch k-means, batch SOM for
  FlowSOM, subsampled GMM fitting and marker tests).

## Sources

* Creighton DoIT, High Performance Computing: https://doit.creighton.edu/HighPerformanceComputing
* Creighton RaD Lab / University Libraries partnership: https://culibraries.creighton.edu/blog/Creighton-Universitys-Radlab-Joins-The-University-Libraries
* Great Plains Network Cyberinfrastructure Program (names Creighton among institutions with HCC agreements): https://www.greatplains.net/community-initiatives/cyberinfrastructure-program/
* HCC home and new user request: https://hcc.unl.edu/ , https://hcc.unl.edu/new-user-request , https://hcc.unl.edu/priority-access-pricing
* HCC AI resources (Swan, PLUMAGE): https://hcc.unl.edu/node/325
* HCC newsletters (ACCESS allocations free for US researchers): https://news.hcc.unl.edu/publications/hcc-newsletter-december-2025 , https://news.hcc.unl.edu/publications/hcc-newsletter-april-2026
* GP-ENGINE: https://gp-engine.umsystem.edu/ , https://hcc.unl.edu/gp-engine-migrating-aiml-workflows-nautilus
* NRP Nautilus access: https://www.caida.org/projects/nids/how-to/access-nrp/ , https://docs.itrss.umsystem.edu/pub/hpc/nautilus/20260501-workshop
