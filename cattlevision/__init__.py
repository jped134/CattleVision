"""CattleVision: Individual cow identification using computer vision."""

__version__ = "0.1.0"

# Lazy imports — torch/torchvision are only required for the identification
# pipeline; behavioral analysis modules work with numpy/cv2/sklearn alone.
try:
    from cattlevision.pipeline.identify import CowIdentifier
    __all__ = ["CowIdentifier", "BehavioralMonitor"]
except ModuleNotFoundError:
    __all__ = ["BehavioralMonitor"]

from cattlevision.behavioral.monitor import BehavioralMonitor
