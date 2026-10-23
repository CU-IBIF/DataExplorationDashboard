"""Command line: serve the dashboard, run pipelines headless (on HPC/cloud), manage jobs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser("featurescope", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the dashboard")
    s.add_argument("--workspace", default="projects", help="directory holding projects")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--root-path", default="", help="URL prefix when behind a proxy (Open OnDemand)")
    s.add_argument("--open", action="store_true", help="open the dashboard in the default browser")

    t = sub.add_parser("selftest", help="end-to-end check of this installation (pipeline + web server)")
    t.add_argument("--keep", help="directory to keep the test project in")

    r = sub.add_parser("run", help="run a pipeline on a project (used on cluster/cloud nodes)")
    r.add_argument("project")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--pipeline", help="name of a saved pipeline")
    g.add_argument("--pipeline-file", help="path to pipeline JSON/YAML")
    r.add_argument("--device", default="auto")
    r.add_argument("--force", action="store_true")
    r.add_argument("--export", action="store_true", help="also write results/<pipeline>.h5ad")

    c = sub.add_parser("create", help="create a project from a table or .h5ad")
    c.add_argument("project")
    c.add_argument("source", help="CSV/TSV/Parquet/h5ad, or 'demo' for synthetic data")
    c.add_argument("--name")
    c.add_argument("--batch")
    c.add_argument("--spatial", nargs=2)
    c.add_argument("--n", type=int, default=20000, help="objects for the demo dataset")

    pl = sub.add_parser("plan", help="show which steps are cached")
    pl.add_argument("project")
    pl.add_argument("--pipeline", required=True)

    sb = sub.add_parser("submit", help="submit a pipeline to a compute profile")
    sb.add_argument("project")
    sb.add_argument("--pipeline", required=True)
    sb.add_argument("--profile", required=True)
    sb.add_argument("--dry-run", action="store_true", help="write scripts and print commands only")
    sb.add_argument("--wait", action="store_true", help="poll until finished, then fetch results")

    st = sub.add_parser("jobs", help="list / refresh remote jobs")
    st.add_argument("project")
    st.add_argument("--refresh", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if a.cmd == "serve":
        import uvicorn
        from .server.app import create_app
        url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}{a.root_path}/"
        print(f"FeatureScope on {url}  (workspace: {Path(a.workspace).resolve()})")
        if a.open:
            import threading
            import webbrowser
            threading.Timer(1.5, webbrowser.open, args=(url,)).start()
        # pure-Python asyncio/h11 stack: identical on every OS and inside the frozen .exe
        uvicorn.run(create_app(a.workspace), host=a.host, port=a.port, root_path=a.root_path, log_level="warning",
                    loop="asyncio", http="h11")

    elif a.cmd == "selftest":
        from .selftest import run
        return run(a.keep)

    elif a.cmd == "create":
        from .project import Project
        if a.source == "demo":
            from .synthetic import make_dataset
            src = make_dataset(a.n)
        else:
            src = a.source
        kw = {k: v for k, v in {"batch": a.batch, "spatial": a.spatial}.items() if v}
        p = Project.create(a.project, src, name=a.name, **kw)
        print(json.dumps(p.info(), indent=1))

    elif a.cmd == "run":
        from .pipeline.engine import Context
        from .pipeline.spec import Pipeline
        from .project import Project
        p = Project(a.project)
        pipe = p.load_pipeline(a.pipeline) if a.pipeline else Pipeline.load(a.pipeline_file)
        ctx = Context(device=a.device, progress=lambda m, f: logging.info("[%3d%%] %s", int(100 * f), m))
        t0 = time.time()
        adata, rep = p.run(pipe, ctx, force=a.force)
        for r in rep:
            print(f"{r['status']:9s} {r['op']:24s} {r.get('seconds', 0):8.2f}s  {r.get('log', '')}")
        if a.export:
            print("exported", p.export(adata, pipe.name))
        print(f"done in {time.time() - t0:.1f}s; result {adata.n_obs} x {adata.n_vars}")

    elif a.cmd == "plan":
        from .project import Project
        p = Project(a.project)
        for r in p.plan(p.load_pipeline(a.pipeline)):
            print(f"{r['status']:9s} {r['op']:24s} {r['key']}")

    elif a.cmd == "submit":
        from .compute import executors as cx
        from .project import Project
        p = Project(a.project)
        ex = cx.get_executor(a.profile, p.root, dry_run=a.dry_run)
        job = ex.submit(p.load_pipeline(a.pipeline).to_dict())
        print(f"job {job.id}: {job.status} remote_id={job.remote_id}")
        print(f"scripts and command log: {job.path(p.root)}")
        while a.wait and not a.dry_run:
            time.sleep(30)
            s = ex.status(job)
            print(time.strftime("%H:%M:%S"), s)
            if s in cx.TERMINAL:
                if s == "COMPLETED":
                    ex.fetch(job)
                    job.status = "fetched"
                    print("results fetched")
                else:
                    job.status = s
                job.save(p.root)
                break

    elif a.cmd == "jobs":
        from .compute import executors as cx
        from .project import Project
        p = Project(a.project)
        for job in cx.Job.list(p.root):
            if a.refresh and job.status not in ("fetched", *cx.TERMINAL):
                ex = cx.get_executor(job.profile, p.root, dry_run=job.dry_run)
                job.status = ex.status(job)
                if job.status == "COMPLETED":
                    ex.fetch(job)
                    job.status = "fetched"
                job.save(p.root)
            print(f"{job.id}  {job.profile:16s} {job.pipeline:20s} {job.status:10s} {job.remote_id}")


if __name__ == "__main__":
    sys.exit(main())
