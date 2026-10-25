
from .dynamic import DynamicAlignmentResult, align_local_distances
from .metric import DiffIPMetric
from .preprocessing import ProjectionResult, project_sequences
from .reversion import fit_step_alignment
from .types import DiffIPResult, StepAlignmentResult

def __getattr__(name: str):
    # lazily expose the torch/transformers-dependent LLM collector so that
    # importing diffip stays light when the optional extras are not installed
    if name in ("LLMRepresentationCollector", "CollectedRepresentations"):
        from . import llm as _llm

        return getattr(_llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffIPMetric",
    "DiffIPResult",
    "DynamicAlignmentResult",
    "ProjectionResult",
    "StepAlignmentResult",
    "align_local_distances",
    "fit_step_alignment",
    "project_sequences",
    "LLMRepresentationCollector",
    "CollectedRepresentations",
]

__version__ = "0.1.0"
