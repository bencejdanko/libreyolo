"""YOLO26 Trainer for LibreYOLO."""

import torch
from typing import Dict, List, Type

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig
from libreyolo.training.freezing import FreezeGroup
from ...training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from ..yolo9.transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset
from .config import YOLO26Config


class YOLO26Trainer(BaseTrainer):
    """YOLO26-specific trainer."""

    artifact_model_families = ("yolo26",)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLO26Config

    def get_model_family(self) -> str:
        return "yolo26"

    def get_model_tag(self) -> str:
        return f"YOLO26-{self.config.size}"

    def get_freeze_groups(self) -> List[FreezeGroup]:
        model = self.model
        backbone = getattr(model, "backbone", None)
        neck = getattr(model, "neck", None)
        head = getattr(model, "head", None)
        groups: List[FreezeGroup] = []
        if backbone is not None:
            for name in (
                "conv0",
                "conv1",
                "elan1",
                "down2",
                "elan2",
                "down3",
                "elan3",
                "down4",
                "elan4",
                "spp",
            ):
                module = getattr(backbone, name, None)
                if module is not None:
                    groups.append((f"backbone.{name}", module))
        if neck is not None:
            for name in (
                "elan_up1",
                "elan_up2",
                "down1",
                "elan_down1",
                "down2",
                "elan_down2",
            ):
                module = getattr(neck, name, None)
                if module is not None:
                    groups.append((f"neck.{name}", module))
        if head is not None:
            groups.append(("head", head))
        return groups or super().get_freeze_groups()

    def create_transforms(self):
        preproc = YOLO9TrainTransform(
            max_labels=100,
            flip_prob=self.config.flip_prob,
            vertical_flip_prob=0.0,
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, YOLO9MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = self.config.scheduler
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        elif scheduler_name in ("cos", "warmcos"):
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        return {
            "box": _scalar(outputs.get("box", 0)),
            "cls": _scalar(outputs.get("cls", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        # Separate parameters into norm weights (pg0), conv/linear weights (pg1), and biases (pg2)
        pg0, pg1, pg2 = [], [], []
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)

        lr = self.effective_lr
        param_groups = []
        if pg0:
            param_groups.append({"params": pg0, "lr": lr})
        if pg1:
            param_groups.append(
                {"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay}
            )
        if pg2:
            param_groups.append({"params": pg2, "lr": lr})

        from .optimizer import MuSGD
        optimizer = MuSGD(
            param_groups,
            lr=lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        return optimizer

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        # Pass epoch and max_epochs down to the model / loss function for ProgLoss
        return self.model(
            imgs,
            targets=targets,
            epoch=self.epoch,
            max_epochs=self.config.epochs
        )
