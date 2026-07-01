"""Training config for YOLO26."""

from dataclasses import dataclass

from ...training.config import YOLO9Config


@dataclass(kw_only=True)
class YOLO26Config(YOLO9Config):
    """YOLO26 uses the same training defaults as YOLOv9 with a custom name."""

    name: str = "yolo26_exp"
