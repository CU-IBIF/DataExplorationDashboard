"""FeatureScope: interactive pipelines for object-by-feature data.

Rows are objects (cells, nuclei, organelles, spots, particles); columns are
features (marker intensities, morphology, gene counts, ...).  Everything is
held in an :class:`anndata.AnnData` so results interoperate with scanpy,
squidpy, scvi-tools and rapids-singlecell.
"""

__version__ = "0.1.0"

from .data import import_table, read_any  # noqa: F401
from .project import Project  # noqa: F401
from .pipeline.spec import Pipeline, Step  # noqa: F401
