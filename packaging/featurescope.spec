# PyInstaller spec for the FeatureScope desktop app (one folder containing FeatureScope.exe).
#   pyinstaller packaging/featurescope.spec --noconfirm
# A one-folder build starts in seconds; a one-file .exe would unpack ~1 GB to a temp dir on
# every launch. The Inno Setup installer (packaging/installer.iss) wraps this folder in a Setup.exe.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
INCLUDE_TORCH = os.environ.get("FEATURESCOPE_NO_TORCH", "0") != "1"

hidden = []
for pkg in ("featurescope", "uvicorn", "scanpy", "anndata", "umap", "pynndescent", "leidenalg", "igraph",
            "sklearn", "h5py", "numba", "fastapi", "starlette", "multipart", "python_multipart"):
    try:
        hidden += collect_submodules(pkg, filter=lambda m: ".tests" not in m and ".testing" not in m)
    except Exception as e:  # optional package absent
        print(f"[spec] skip submodules of {pkg}: {e}")

datas = collect_data_files("featurescope")            # web UI, vendored Plotly, job templates
for pkg in ("scanpy", "anndata", "umap", "pynndescent", "sklearn", "igraph", "leidenalg", "legacy_api_wrap"):
    try:
        datas += collect_data_files(pkg)
    except Exception as e:
        print(f"[spec] skip data of {pkg}: {e}")

# Packages that look up their own version through importlib.metadata at import time.
for dist in ("featurescope", "scanpy", "anndata", "numpy", "scipy", "pandas", "scikit-learn", "numba", "llvmlite",
             "umap-learn", "pynndescent", "h5py", "igraph", "leidenalg", "matplotlib", "fastapi", "starlette",
             "uvicorn", "pydantic", "legacy-api-wrap", "session-info2", "natsort", "joblib", "threadpoolctl",
             "packaging", "networkx", "statsmodels", "seaborn", "patsy", "array-api-compat", "fast-array-utils",
             "zarr", "numcodecs", "tqdm", "pyarrow", "torch"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

excludes = ["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "jupyter", "notebook", "pytest",
            "torch.utils.tensorboard", "tensorboard", "sphinx", "dask.distributed"]
if not INCLUDE_TORCH:
    excludes += ["torch", "torchvision", "torchaudio"]

a = Analysis(
    [os.path.join(SPEC_DIR, "launch.py")],
    pathex=[os.path.dirname(SPEC_DIR)],   # repo root, so editable installs also resolve
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    # scanpy calls inspect.getsource() at import time and numba's on-disk JIT cache (UMAP,
    # pynndescent) locates functions by source file, so these ship their .py files too.
    module_collection_mode={pkg: "pyz+py" for pkg in
                            ("scanpy", "anndata", "umap", "pynndescent", "numba", "legacy_api_wrap", "featurescope")},
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
icon = os.path.join(SPEC_DIR, "featurescope.ico")
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FeatureScope",
    console=True,            # the console window shows the URL and logs; closing it stops the server
    icon=icon if os.path.exists(icon) else None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="FeatureScope")
