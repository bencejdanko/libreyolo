"""LibreYOLO26 wrapper: detect, segment, pose, obb, and classify tasks."""

from __future__ import annotations
from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseModel
from ...tasks import normalize_task
from ...validation.preprocessors import YOLO9E2EValPreprocessor
from .nn import (
    LibreYOLO26Model,
    LibreYOLO26SegModel,
    LibreYOLO26PoseModel,
    LibreYOLO26OBBModel,
    LibreYOLO26ClsModel,
)
from .utils import postprocess as _postprocess
from .utils import preprocess_numpy as _preprocess_numpy
from .config import YOLO26Config


class LibreYOLO26(BaseModel):
    """LibreYOLO26 multi-task model implementation."""

    FAMILY = "yolo26"
    FILENAME_PREFIX = "LibreYOLO26"
    
    # Input sizes
    INPUT_SIZES = {"n": 640, "s": 640, "m": 640, "l": 640, "x": 640}
    CLS_INPUT_SIZES = {"n": 224, "s": 224, "m": 224, "l": 224, "x": 224}
    
    SUPPORTED_TASKS = ("detect", "segment", "pose", "obb", "classify")
    DEFAULT_TASK = "detect"
    
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "segment": INPUT_SIZES,
        "pose": INPUT_SIZES,
        "obb": INPUT_SIZES,
        "classify": CLS_INPUT_SIZES,
    }
    
    TRAIN_CONFIG = YOLO26Config
    val_preprocessor_class = YOLO9E2EValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Check for presence of NMS-free one-to-one keys
        has_e2e = any("one2one_cv2" in k or "one2one_cv3" in k for k in weights_dict)
        # Check if this is a classification checkpoint
        has_cls = "fc.weight" in weights_dict and "backbone.conv0.conv.weight" in weights_dict
        
        if not (has_e2e or has_cls):
            return False
            
        # Standard E2E has DFL (dfl.project), YOLO26 does not
        if any("dfl.project" in k for k in weights_dict):
            return False
            
        # Verify box regression output dimension is 4 (not 64 for DFL)
        if has_e2e:
            for k in weights_dict:
                if "one2one_cv2.0.2.weight" in k:
                    if weights_dict[k].shape[0] == 4:
                        return True
            return False
            
        return True

    @classmethod
    def is_pose_state_dict(cls, weights_dict: dict) -> bool:
        return any("one2one_cv5" in k for k in weights_dict)

    @classmethod
    def is_seg_state_dict(cls, weights_dict: dict) -> bool:
        return any("one2one_cv4" in k for k in weights_dict)

    @classmethod
    def is_obb_state_dict(cls, weights_dict: dict) -> bool:
        return any("one2one_cv6" in k for k in weights_dict)

    @classmethod
    def is_classify_state_dict(cls, weights_dict: dict) -> bool:
        return "fc.weight" in weights_dict and "fc.bias" in weights_dict

    @classmethod
    def detect_task_from_state_dict(cls, weights_dict: dict) -> Optional[str]:
        if cls.is_pose_state_dict(weights_dict):
            return "pose"
        if cls.is_seg_state_dict(weights_dict):
            return "segment"
        if cls.is_obb_state_dict(weights_dict):
            return "obb"
        if cls.is_classify_state_dict(weights_dict):
            return "classify"
        return None

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> Optional[str]:
        return cls.detect_task_from_state_dict(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "backbone.conv0.conv.weight"
        if key not in weights_dict:
            return None
        first_channel = weights_dict[key].shape[0]
        if first_channel == 16:
            return "n"
        if first_channel == 32:
            secondary_key = "backbone.elan2.cv1.conv.weight"
            if secondary_key in weights_dict:
                mid_channel = weights_dict[secondary_key].shape[0]
                if mid_channel == 128:
                    return "s"
                elif mid_channel == 192:
                    return "m"
            return "s"
        if first_channel == 64:
            secondary_key = "backbone.elan2.cv1.conv.weight"
            if secondary_key in weights_dict:
                mid_channel = weights_dict[secondary_key].shape[0]
                if mid_channel == 256:
                    return "l"
                elif mid_channel == 384:
                    return "x"
            return "l"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        if "fc.weight" in weights_dict:
            return int(weights_dict["fc.weight"].shape[0])
            
        for k in weights_dict:
            if "one2one_cv3.0.2.weight" in k:
                return int(weights_dict[k].shape[0])
        return None

    def _init_model(self) -> nn.Module:
        if self.task == "pose":
            return LibreYOLO26PoseModel(size=self.size, nc=1)
        if self.task == "segment":
            return LibreYOLO26SegModel(size=self.size, nc=self.nb_classes)
        if self.task == "obb":
            return LibreYOLO26OBBModel(size=self.size, nc=self.nb_classes)
        if self.task == "classify":
            return LibreYOLO26ClsModel(size=self.size, nc=self.nb_classes)
        return LibreYOLO26Model(size=self.size, nc=self.nb_classes)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        if self.task == "classify":
            return {
                "backbone": self.model.backbone,
                "fc": self.model.fc,
            }
        return {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return _preprocess_numpy

    def _preprocess(self, image, *, color_format=None, **kwargs):
        # We reuse the shared letterbox preprocessing for YOLO9
        from ..yolo9.model import LibreYOLO9
        return LibreYOLO9._preprocess(self, image, color_format=color_format, **kwargs)

    def _forward(self, x: torch.Tensor, epoch=None, max_epochs=None) -> Any:
        if self.training:
            return self.model(x, epoch=epoch, max_epochs=max_epochs)
        return self.model(x)

    def _postprocess(self, raw, conf_thres: float, iou_thres: float, **kwargs):
        return _postprocess(raw, conf_thres, iou_thres, **kwargs)
