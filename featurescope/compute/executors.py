"""Offloading pipeline runs to institutional HPC (Slurm), Kubernetes (NRP Nautilus) or cloud batch.

Every backend follows the same four verbs, so the dashboard treats them alike:

    stage   copy the project's inputs (base data, pipeline, gates, existing cache) to the target
    submit  start ``featurescope run`` there (GPU if requested)
    status  poll the scheduler
    fetch   copy ``cache/`` back; results then load locally with zero recomputation

Credentials are never stored by FeatureScope: it shells out to ``ssh``/``rsync``,
``kubectl``, ``aws`` or ``gcloud`` and uses whatever keys / tokens / SSO the user
has configured for those tools (SSH ControlMaster makes Duo/MFA a once-per-day prompt).

Profiles live in ``~/.featurescope/compute.yaml`` and/or ``<project>/compute.yaml``
(see ``deploy/compute.example.yaml``).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template

import yaml

TEMPLATES = Path(__file__).parent / "templates"
STAGE_ITEMS = ["project.json", "gates.json", "data", "pipelines", "cache"]


# ----------------------------------------------------------------------------- profiles

def load_profiles(project_root: str | Path | None = None) -> dict:
    profiles = {"local": {"type": "local", "description": "This computer"}}
    paths = [Path(os.environ.get("FEATURESCOPE_COMPUTE", Path.home() / ".featurescope" / "compute.yaml"))]
    if project_root:
        paths.append(Path(project_root) / "compute.yaml")
    for p in paths:
        if p.exists():
            profiles.update((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("profiles", {}))
    return profiles


# ----------------------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    profile: str
    pipeline: str
    status: str = "created"             # created|staged|submitted|PENDING|RUNNING|COMPLETED|FAILED|fetched|...
    remote_id: str | None = None
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    remote_path: str | None = None
    history: list = field(default_factory=list)
    dry_run: bool = False

    def path(self, project_root: Path) -> Path:
        return Path(project_root) / "jobs" / self.id

    def save(self, project_root: Path) -> None:
        d = self.path(project_root)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, project_root: Path, job_id: str) -> "Job":
        return cls(**json.loads((Path(project_root) / "jobs" / job_id / "job.json").read_text(encoding="utf-8")))

    @classmethod
    def list(cls, project_root: Path) -> list["Job"]:
        out = []
        for p in sorted((Path(project_root) / "jobs").glob("*/job.json")):
            try:
                out.append(cls(**json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                pass
        return sorted(out, key=lambda j: j.created, reverse=True)

    def log(self, msg: str) -> None:
        self.history.append(f"{time.strftime('%H:%M:%S')} {msg}")


class Runner:
    """Runs (or, in dry-run mode, records) shell commands."""

    def __init__(self, dry_run: bool = False, log_file: Path | None = None):
        self.dry_run, self.log_file, self.commands = dry_run, log_file, []

    def __call__(self, cmd: list[str], input: str | None = None, check: bool = True, timeout: int = 3600) -> str:
        self.commands.append(cmd)
        line = " ".join(shlex.quote(c) for c in cmd)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(f"$ {line}\n")
        if self.dry_run:
            return ""
        r = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(r.stdout + r.stderr)
        if check and r.returncode != 0:
            raise RuntimeError(f"command failed ({r.returncode}): {line}\n{r.stderr.strip()[-2000:]}")
        return r.stdout


def _render(template: str, **values) -> str:
    return Template((TEMPLATES / template).read_text(encoding="utf-8")).safe_substitute(**values)


def _run_cmd(remote_project: str, pipeline_file: str, prof: dict) -> str:
    dev = "cuda" if prof.get("gpus") or prof.get("gres") else "auto"
    return f"featurescope run {shlex.quote(remote_project)} --pipeline-file {shlex.quote(pipeline_file)} --device {dev} --export"


# ----------------------------------------------------------------------------- executors

class Executor:
    def __init__(self, name: str, profile: dict, project_root: Path, dry_run: bool = False):
        self.name, self.prof, self.root = name, profile, Path(project_root)
        self.dry_run = dry_run or profile.get("dry_run", False)

    def runner(self, job: Job) -> Runner:
        job.path(self.root).mkdir(parents=True, exist_ok=True)
        return Runner(self.dry_run, job.path(self.root) / "commands.log")

    def new_job(self, pipeline: dict) -> Job:
        job = Job(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], profile=self.name,
                  pipeline=pipeline["name"], dry_run=self.dry_run)
        job.path(self.root).mkdir(parents=True, exist_ok=True)
        (job.path(self.root) / "pipeline.json").write_text(json.dumps(pipeline, indent=1), encoding="utf-8")
        return job

    def submit(self, pipeline: dict) -> Job:
        raise NotImplementedError

    def status(self, job: Job) -> str:
        raise NotImplementedError

    def fetch(self, job: Job) -> None:
        raise NotImplementedError

    def cancel(self, job: Job) -> None:
        raise NotImplementedError


class SlurmExecutor(Executor):
    """Slurm cluster reachable over SSH (Creighton research computing, UNL HCC Swan, ACCESS sites)."""

    def _ssh(self) -> list[str]:
        p = self.prof
        target = f"{p['user']}@{p['host']}" if p.get("user") else p["host"]
        opts = ["-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o",
                f"ControlPath={Path.home()}/.ssh/fs-%r@%h:%p", "-o", "ControlPersist=8h"]
        return ["ssh", *opts, *p.get("ssh_opts", []), target]

    def _target(self) -> str:
        p = self.prof
        return f"{p['user']}@{p['host']}" if p.get("user") else p["host"]

    def remote_project(self) -> str:
        meta = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        root = self.prof.get("remote_root", "~/featurescope").replace("{user}", self.prof.get("user", "$USER"))
        return f"{root.rstrip('/')}/{meta['base_key']}"   # same data -> same remote dir, uploads are incremental

    def submit(self, pipeline: dict) -> Job:
        job = self.new_job(pipeline)
        run = self.runner(job)
        rp = self.remote_project()
        job.remote_path = rp
        rsh = " ".join(shlex.quote(c) for c in self._ssh()[:-1])
        run([*self._ssh(), f"mkdir -p {rp}/jobs/{job.id}"])
        # rsync is incremental: base.h5ad uploads once per dataset; only new cache entries move afterwards
        run(["rsync", "-az", "--partial", "-e", rsh, *[str(self.root / i) for i in STAGE_ITEMS if (self.root / i).exists()],
             f"{self._target()}:{rp}/"])
        p = self.prof
        script = _render("slurm.sbatch", job_name=f"fs-{job.id}", partition=p.get("partition", ""),
                         cpus=p.get("cpus", 8), mem=p.get("mem", "64G"), time=p.get("time", "04:00:00"),
                         gres_line=f"#SBATCH --gres={p['gres']}" if p.get("gres") else "",
                         account_line=f"#SBATCH --account={p['account']}" if p.get("account") else "",
                         extra_sbatch="\n".join(f"#SBATCH {x}" for x in p.get("sbatch_extra", [])),
                         workdir=f"{rp}/jobs/{job.id}", env_setup=p.get("env_setup", ""),
                         run_cmd=self._wrap(_run_cmd(rp, f"{rp}/jobs/{job.id}/pipeline.json", p)))
        (job.path(self.root) / "job.sbatch").write_text(script, encoding="utf-8")
        run(["rsync", "-az", "-e", rsh, str(job.path(self.root) / "pipeline.json"),
             str(job.path(self.root) / "job.sbatch"), f"{self._target()}:{rp}/jobs/{job.id}/"])
        out = run([*self._ssh(), f"cd {rp}/jobs/{job.id} && sbatch --parsable job.sbatch"])
        job.remote_id = out.strip().split(";")[0] or ("DRYRUN" if self.dry_run else None)
        job.status = "submitted"
        job.log(f"submitted slurm job {job.remote_id} on {p['host']}")
        job.save(self.root)
        return job

    def _wrap(self, cmd: str) -> str:
        """Run inside an Apptainer image if configured (recommended), else in the activated env."""
        img = self.prof.get("container")
        if img:
            nv = "--nv " if self.prof.get("gres") else ""
            return f"apptainer exec {nv}--bind $PWD/../..:$PWD/../.. {shlex.quote(img)} {cmd}"
        return cmd

    def status(self, job: Job) -> str:
        if self.dry_run:
            return "COMPLETED"
        run = self.runner(job)
        out = run([*self._ssh(), f"squeue -h -j {job.remote_id} -o %T 2>/dev/null || true"], check=False).strip()
        if not out:
            out = run([*self._ssh(), f"sacct -n -X -j {job.remote_id} -o State 2>/dev/null | head -1"], check=False).strip()
        return (out.split() or ["UNKNOWN"])[0]

    def fetch(self, job: Job) -> None:
        run = self.runner(job)
        rsh = " ".join(shlex.quote(c) for c in self._ssh()[:-1])
        run(["rsync", "-az", "-e", rsh, f"{self._target()}:{job.remote_path}/cache/", str(self.root / "cache") + "/"])
        run(["rsync", "-az", "-e", rsh, f"{self._target()}:{job.remote_path}/jobs/{job.id}/", str(job.path(self.root)) + "/"],
            check=False)

    def cancel(self, job: Job) -> None:
        self.runner(job)([*self._ssh(), f"scancel {job.remote_id}"])


class _ObjectStoreMixin:
    """Stage through S3-compatible storage (AWS S3, NRP Ceph S3, MinIO) or GCS."""

    def store_uri(self) -> str:
        meta = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        return f"{self.prof['storage'].rstrip('/')}/{meta['base_key']}"

    def _sync_cmd(self, src: str, dst: str) -> list[str]:
        if self.prof["storage"].startswith("gs://"):
            return ["gsutil", "-m", "rsync", "-r", src, dst]
        cmd = ["aws", "s3", "sync", src, dst, "--only-show-errors"]
        if self.prof.get("s3_endpoint"):
            cmd += ["--endpoint-url", self.prof["s3_endpoint"]]
        return cmd

    def _copy_cmd(self, src: str, dst: str) -> list[str]:
        if self.prof["storage"].startswith("gs://"):
            return ["gsutil", "cp", src, dst]
        cmd = ["aws", "s3", "cp", src, dst, "--only-show-errors"]
        if self.prof.get("s3_endpoint"):
            cmd += ["--endpoint-url", self.prof["s3_endpoint"]]
        return cmd

    def stage(self, job: Job, run: Runner) -> str:
        uri = self.store_uri()
        for item in STAGE_ITEMS:
            p = self.root / item
            if p.is_dir():
                run(self._sync_cmd(str(p), f"{uri}/{item}"))
            elif p.exists():
                run(self._copy_cmd(str(p), f"{uri}/{item}"))
        job_uri = f"{uri}/jobs/{job.id}"
        run(self._sync_cmd(str(job.path(self.root)), job_uri))
        job.remote_path = uri
        return uri

    def container_script(self, uri: str, job: Job) -> str:
        """Shell run inside the container: pull project, run, push cache back."""
        ep = f" --endpoint-url {self.prof['s3_endpoint']}" if self.prof.get("s3_endpoint") else ""
        if uri.startswith("gs://"):
            pull, push = f"gsutil -m rsync -r {uri} /work/project", f"gsutil -m rsync -r /work/project/cache {uri}/cache"
        else:
            pull = f"aws s3 sync {uri} /work/project --only-show-errors{ep}"
            push = (f"aws s3 sync /work/project/cache {uri}/cache --only-show-errors{ep} && "
                    f"aws s3 sync /work/project/results {uri}/results --only-show-errors{ep}")
        run = _run_cmd("/work/project", f"/work/project/jobs/{job.id}/pipeline.json", self.prof)
        return f"set -euo pipefail; mkdir -p /work/project; {pull}; {run}; {push}"

    def fetch(self, job: Job) -> None:
        run = self.runner(job)
        run(self._sync_cmd(f"{job.remote_path}/cache", str(self.root / "cache")))


class KubernetesExecutor(_ObjectStoreMixin, Executor):
    """Kubernetes Job (NRP Nautilus via GP-ENGINE, or any institutional k8s with GPUs)."""

    def _kubectl(self) -> list[str]:
        cmd = ["kubectl"]
        if self.prof.get("context"):
            cmd += ["--context", self.prof["context"]]
        return cmd + ["-n", self.prof["namespace"]]

    def submit(self, pipeline: dict) -> Job:
        job = self.new_job(pipeline)
        run = self.runner(job)
        uri = self.stage(job, run)
        p = self.prof
        name = f"featurescope-{job.id}".lower().replace("_", "-")[:63]
        gpu = int(p.get("gpus", 0))
        manifest = _render("k8s-job.yaml", name=name, namespace=p["namespace"], image=p["image"],
                           cpu=p.get("cpus", 8), memory=p.get("memory", "64Gi"),
                           gpu_limits=f'\n              nvidia.com/gpu: "{gpu}"' if gpu else "",
                           secret=p.get("s3_secret", "s3-credentials"),
                           script=json.dumps(self.container_script(uri, job)),
                           node_selector=json.dumps(p.get("node_selector", {})))
        (job.path(self.root) / "job.yaml").write_text(manifest, encoding="utf-8")
        run([*self._kubectl(), "apply", "-f", str(job.path(self.root) / "job.yaml")])
        job.remote_id, job.status = name, "submitted"
        job.log(f"submitted k8s job {name} in namespace {p['namespace']}")
        job.save(self.root)
        return job

    def status(self, job: Job) -> str:
        if self.dry_run:
            return "COMPLETED"
        out = self.runner(job)([*self._kubectl(), "get", "job", job.remote_id, "-o", "json"], check=False)
        try:
            st = json.loads(out)["status"]
        except Exception:
            return "UNKNOWN"
        if st.get("succeeded"):
            return "COMPLETED"
        if st.get("failed"):
            return "FAILED"
        return "RUNNING" if st.get("active") else "PENDING"

    def cancel(self, job: Job) -> None:
        self.runner(job)([*self._kubectl(), "delete", "job", job.remote_id])


class AwsBatchExecutor(_ObjectStoreMixin, Executor):
    """AWS Batch (CPU or GPU compute environment) staged through S3."""

    def submit(self, pipeline: dict) -> Job:
        job = self.new_job(pipeline)
        run = self.runner(job)
        uri = self.stage(job, run)
        p = self.prof
        overrides = {"command": ["bash", "-c", self.container_script(uri, job)],
                     "resourceRequirements": [{"type": "VCPU", "value": str(p.get("cpus", 8))},
                                              {"type": "MEMORY", "value": str(p.get("memory_mib", 61440))}]
                     + ([{"type": "GPU", "value": str(p["gpus"])}] if p.get("gpus") else [])}
        out = run(["aws", "batch", "submit-job", "--job-name", f"featurescope-{job.id}", "--job-queue", p["job_queue"],
                   "--job-definition", p["job_definition"], "--container-overrides", json.dumps(overrides),
                   *(["--region", p["region"]] if p.get("region") else [])])
        job.remote_id = json.loads(out)["jobId"] if out else "DRYRUN"
        job.status = "submitted"
        job.save(self.root)
        return job

    def status(self, job: Job) -> str:
        if self.dry_run:
            return "COMPLETED"
        out = self.runner(job)(["aws", "batch", "describe-jobs", "--jobs", job.remote_id], check=False)
        try:
            s = json.loads(out)["jobs"][0]["status"]
        except Exception:
            return "UNKNOWN"
        return {"SUCCEEDED": "COMPLETED"}.get(s, s)

    def cancel(self, job: Job) -> None:
        self.runner(job)(["aws", "batch", "terminate-job", "--job-id", job.remote_id, "--reason", "user"])


class GcpBatchExecutor(_ObjectStoreMixin, Executor):
    """Google Cloud Batch staged through GCS."""

    def submit(self, pipeline: dict) -> Job:
        job = self.new_job(pipeline)
        run = self.runner(job)
        uri = self.stage(job, run)
        p = self.prof
        spec = {"taskGroups": [{"taskSpec": {
                    "runnables": [{"container": {"imageUri": p["image"], "entrypoint": "bash",
                                                 "commands": ["-c", self.container_script(uri, job)]}}],
                    "computeResource": {"cpuMilli": int(p.get("cpus", 8)) * 1000, "memoryMib": p.get("memory_mib", 61440)},
                    "maxRetryCount": 0}}],
                "allocationPolicy": {"instances": [{"installGpuDrivers": bool(p.get("gpus")), "policy": {
                    "machineType": p.get("machine_type", "n1-highmem-8"),
                    **({"accelerators": [{"type": p.get("gpu_type", "nvidia-tesla-t4"), "count": p["gpus"]}]}
                       if p.get("gpus") else {})}}]},
                "logsPolicy": {"destination": "CLOUD_LOGGING"}}
        f = job.path(self.root) / "gcp-batch.json"
        f.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        name = f"featurescope-{job.id}".lower()
        run(["gcloud", "batch", "jobs", "submit", name, "--location", p.get("region", "us-central1"), "--config", str(f)])
        job.remote_id, job.status = name, "submitted"
        job.save(self.root)
        return job

    def status(self, job: Job) -> str:
        if self.dry_run:
            return "COMPLETED"
        out = self.runner(job)(["gcloud", "batch", "jobs", "describe", job.remote_id, "--location",
                                self.prof.get("region", "us-central1"), "--format", "value(status.state)"], check=False)
        s = out.strip() or "UNKNOWN"
        return {"SUCCEEDED": "COMPLETED", "QUEUED": "PENDING", "SCHEDULED": "PENDING"}.get(s, s)

    def cancel(self, job: Job) -> None:
        self.runner(job)(["gcloud", "batch", "jobs", "delete", job.remote_id, "--location",
                          self.prof.get("region", "us-central1"), "--quiet"])


EXECUTORS = {"slurm": SlurmExecutor, "kubernetes": KubernetesExecutor, "aws_batch": AwsBatchExecutor,
             "gcp_batch": GcpBatchExecutor}


def get_executor(profile_name: str, project_root: Path, dry_run: bool = False) -> Executor:
    profiles = load_profiles(project_root)
    if profile_name not in profiles:
        raise KeyError(f"unknown compute profile {profile_name!r}; available: {list(profiles)}")
    prof = profiles[profile_name]
    if prof["type"] == "local":
        raise ValueError("local runs go through the dashboard's job queue, not an executor")
    return EXECUTORS[prof["type"]](profile_name, prof, Path(project_root), dry_run)


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}


def recommend(n_obs: int, n_vars: int, pending_ops: list[dict]) -> dict:
    """Rough guidance on where to run the pending steps."""
    heavy = [o["name"] for o in pending_ops if o.get("heavy")]
    gpu = [o["name"] for o in pending_ops if o.get("gpu") and o.get("heavy")]
    gb = n_obs * n_vars * 4 / 1e9
    deep = [o for o in heavy if o in ("vae", "contrastive", "spatial_dgi", "scvi")]
    if n_obs < 200_000 and not deep:
        where, why = "local", "small enough for a laptop/workstation"
    elif n_obs < 1_000_000 and not deep:
        where, why = "local or HPC CPU node", "kNN/UMAP on <1M objects runs in minutes on 8+ cores"
    elif deep and n_obs >= 200_000:
        where, why = "HPC/cloud GPU node", f"deep embeddings ({', '.join(deep)}) are 10-50x faster on a GPU"
    else:
        where, why = "HPC GPU node (rapids-singlecell)", "millions of objects: GPU kNN/UMAP/Leiden"
    return {"where": where, "why": why, "matrix_gb": round(gb, 2), "est_ram_gb": round(gb * 6 + 2, 1),
            "heavy_steps": heavy, "gpu_steps": gpu}
