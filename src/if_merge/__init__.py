"""IF-Merge: model merging with iterative function-space feedback."""

from .config import IFMergeConfig
from .core.merge import IFMergeResult, RoundRecord, if_merge

__version__ = "1.0.0"
__all__ = ["IFMergeConfig", "IFMergeResult", "RoundRecord", "if_merge"]
