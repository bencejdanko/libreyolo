from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import BaseModel
from ...tasks import normalize_task
from ...validation.preprocessors import YOLO9E2EValPreprocessor
from ...training.ddp_spawn import ddp_aware
from ...utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    validate_checkpoint_metadata,
    load_untrusted_torch_file,
)
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

logger = logging.getLogger(__name__)
_TRAIN_DEFAULTS = YOLO26Config()


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
        if self.model.training:
            return self.model(x, epoch=epoch, max_epochs=max_epochs)
        return self.model(x)

    def _postprocess(self, raw, conf_thres: float, iou_thres: float, **kwargs):
        return _postprocess(raw, conf_thres, iou_thres, **kwargs)

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int = _TRAIN_DEFAULTS.imgsz,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        allow_download_scripts: bool = False,
        pretrained: bool | str | Path | None = None,
        callbacks=None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train the YOLO26 model on a dataset."""
        from .trainer import YOLO26Trainer
        from libreyolo.data import load_data_config

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                allow_scripts=allow_download_scripts,
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")

        # If no nc in data.yaml, infer it by counting.
        if yaml_nc is None and yaml_names is not None:
            yaml_nc = len(yaml_names)
        if yaml_nc is not None:
            yaml_nc = int(yaml_nc)

        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        # Apply custom class names from data config
        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if resume and pretrained:
            raise ValueError("pretrained transfer cannot be combined with resume=True.")

        if pretrained:
            transfer_weights: str | Path
            if pretrained is True:
                transfer_weights = self._default_transfer_weights_name()
            else:
                transfer_weights = pretrained
            stats = self._load_transfer_weights(transfer_weights)
            logger.info(
                "Loaded %d transfer tensors from %s; skipped %d incompatible tensors.",
                stats["loaded"],
                transfer_weights,
                stats["skipped"],
            )

        if seed >= 0:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer_kwargs = dict(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            allow_download_scripts=allow_download_scripts,
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )
        trainer = YOLO26Trainer(**trainer_kwargs)

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreYOLO26('path/to/last.pt', size='t'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()

        self._restore_after_training(results)

        return results

    def _default_transfer_weights_name(self) -> str:
        """Return the matching detect checkpoint filename for transfer learning."""
        return f"{self.FILENAME_PREFIX}{self.size}{self.WEIGHT_EXT}"

    def _load_transfer_weights(self, weights: str | Path) -> dict[str, int]:
        """Partially load same-family weights for training initialization."""
        path = Path(self._resolve_weights_path(str(weights)))
        if not path.exists():
            from ...utils.download import download_weights

            download_weights(str(path), self.size)

        if not path.exists():
            raise FileNotFoundError(f"Transfer weights not found at {weights}")

        loaded = load_untrusted_torch_file(
            str(path),
            map_location="cpu",
            context="transfer weights",
        )
        if isinstance(loaded, dict):
            metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
            if metadata_keys & set(loaded):
                metadata_errors = validate_checkpoint_metadata(loaded, strict=False)
                if metadata_errors:
                    raise RuntimeError(
                        "Transfer checkpoint metadata is incomplete: "
                        + "; ".join(metadata_errors)
                    )

            ckpt_family = loaded.get("model_family", "")
            if ckpt_family and ckpt_family != self._get_model_name():
                raise RuntimeError(
                    f"Transfer checkpoint model_family='{ckpt_family}' does not "
                    f"match '{self._get_model_name()}'."
                )

            ckpt_task = loaded.get("task")
            if ckpt_task is not None:
                normalized_ckpt_task = normalize_task(ckpt_task)
                if normalized_ckpt_task != self.task and normalized_ckpt_task != "detect":
                    raise RuntimeError(
                        f"Transfer checkpoint task='{normalized_ckpt_task}' is "
                        f"not compatible with task='{self.task}'."
                    )

            if "model" in loaded:
                state_dict = loaded["model"]
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
            else:
                state_dict = loaded
        else:
            state_dict = loaded

        state_dict = self._prepare_state_dict(self._strip_ddp_prefix(state_dict))
        total_tensors = len(state_dict)

        current = self.model.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        current.update(matched)
        self.model.load_state_dict(current, strict=True)
        self.model.to(self.device)
        return {
            "loaded": len(matched),
            "skipped": max(total_tensors - len(matched), 0),
        }

    def _restore_after_training(self, results: dict) -> None:
        """Reload the saved checkpoint and leave the model ready for inference."""
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = results.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break

        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)

        self.model.to(self.device).eval()
