import json

import yaml

from featurescope import Pipeline
from featurescope.compute import executors as cx


def _profiles(root):
    (root / "compute.yaml").write_text(yaml.safe_dump({"profiles": {
        "hpc": {"type": "slurm", "host": "login.example.edu", "user": "me", "partition": "gpu", "gres": "gpu:1",
                "remote_root": "/scratch/{user}/fs", "container": "/scratch/me/featurescope.sif",
                "env_setup": "module load apptainer"},
        "nrp": {"type": "kubernetes", "namespace": "lab", "image": "ghcr.io/x/featurescope:gpu", "gpus": 1,
                "storage": "s3://bucket/fs", "s3_endpoint": "https://s3.example.org"},
        "aws": {"type": "aws_batch", "job_queue": "q", "job_definition": "d", "storage": "s3://b/fs", "gpus": 1},
        "gcp": {"type": "gcp_batch", "image": "img", "storage": "gs://b/fs"},
    }}), encoding="utf-8")


def test_dry_run_all_backends(project):
    _profiles(project.root)
    pl = Pipeline("p").add("normalize").add("pca")
    project.save_pipeline(pl)
    for name in ("hpc", "nrp", "aws", "gcp"):
        ex = cx.get_executor(name, project.root, dry_run=True)
        job = ex.submit(pl.to_dict())
        assert job.status == "submitted"
        log = (job.path(project.root) / "commands.log").read_text(encoding="utf-8")
        assert log.strip()
        if name == "hpc":
            script = (job.path(project.root) / "job.sbatch").read_text(encoding="utf-8")
            assert "#SBATCH --gres=gpu:1" in script and "apptainer exec --nv" in script
            assert "featurescope run" in script and "rsync" in log and "sbatch --parsable" in log
        if name == "nrp":
            manifest = yaml.safe_load((job.path(project.root) / "job.yaml").read_text(encoding="utf-8"))
            c = manifest["spec"]["template"]["spec"]["containers"][0]
            assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
            assert "featurescope run" in c["args"][0] and "--endpoint-url" in c["args"][0]
        assert ex.status(job) == "COMPLETED"
    jobs = cx.Job.list(project.root)
    assert len(jobs) == 4 and json.loads((jobs[0].path(project.root) / "pipeline.json").read_text(encoding="utf-8"))["name"] == "p"


def test_recommend():
    ops = [{"name": "vae", "heavy": True, "gpu": True}]
    assert "GPU" in cx.recommend(2_000_000, 40, ops)["where"]
    assert cx.recommend(10_000, 40, [])["where"] == "local"


def test_remote_results_load_without_recompute(project, tmp_path):
    """Simulate a cluster: copy project, run there, copy cache back."""
    import shutil
    from featurescope import Project
    pl = Pipeline("p").add("normalize").add("pca", n_comps=4)
    remote = tmp_path / "remote"
    shutil.copytree(project.root, remote)
    Project(remote).run(pl)                                 # "on the cluster"
    shutil.copytree(remote / "cache", project.root / "cache", dirs_exist_ok=True)   # fetch
    _, rep = project.run(pl, cache_only=True)
    assert [r["status"] for r in rep] == ["cached", "cached"]
