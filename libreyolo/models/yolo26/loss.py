"""YOLO26 loss functions for training."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..yolo9.loss import generate_anchors, BCELoss, BoxLoss, calculate_iou


class YOLO26BoxMatcher:
    """
    Small-Target-Aware Label Assignment (STAL) for matching ground truths to anchors.
    """

    def __init__(
        self,
        num_classes: int,
        anchor_grid: Tensor,
        scaler: Tensor,
        topk: int = 10,
        iou_factor: float = 6.0,
        cls_factor: float = 0.5,
    ):
        self.num_classes = num_classes
        self.anchor_grid = anchor_grid
        self.scaler = scaler
        self.topk = topk
        self.iou_factor = iou_factor
        self.cls_factor = cls_factor

    def get_valid_matrix(self, target_bbox: Tensor) -> Tensor:
        """Get valid anchor mask."""
        x_min, y_min, x_max, y_max = target_bbox[:, :, None].unbind(3)
        anchors = self.anchor_grid[None, None]  # (1, 1, anchors, 2)
        anchors_x, anchors_y = anchors.unbind(dim=3)

        x_min_dist, x_max_dist = anchors_x - x_min, x_max - anchors_x
        y_min_dist, y_max_dist = anchors_y - y_min, y_max - anchors_y
        targets_dist = torch.stack(
            (x_min_dist, y_min_dist, x_max_dist, y_max_dist), dim=-1
        )
        targets_dist /= self.scaler[None, None, :, None]

        min_reg_dist = targets_dist.amin(dim=-1)
        # Bounding boxes are valid if the anchor is inside the target box
        return min_reg_dist >= 0

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        predict_cls = predict_cls.transpose(1, 2)  # (B, nc, anchors)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))  # (B, targets, anchors)
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox: Tensor, target_bbox: Tensor) -> Tensor:
        return calculate_iou(target_bbox, predict_bbox, "ciou").clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, grid_mask: Tensor, topk: int = 10) -> Tuple[Tensor, Tensor]:
        masked_target_matrix = grid_mask * target_matrix
        values, indices = masked_target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_mask = topk_targets > 0
        return topk_targets, topk_mask

    def ensure_one_anchor(self, target_matrix: Tensor, topk_mask: Tensor) -> Tensor:
        values, indices = target_matrix.max(dim=-1)
        best_anchor_mask = torch.zeros_like(target_matrix, dtype=torch.bool)
        best_anchor_mask.scatter_(-1, index=indices[..., None], src=~best_anchor_mask)
        matched_anchor_num = torch.sum(topk_mask, dim=-1)
        target_without_anchor = (matched_anchor_num == 0) & (values > 0)
        topk_mask = torch.where(target_without_anchor[..., None], best_anchor_mask, topk_mask)
        return topk_mask

    def filter_duplicates(self, iou_mat: Tensor, topk_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        duplicates = (topk_mask.sum(1, keepdim=True) > 1).repeat([1, topk_mask.size(1), 1])
        masked_iou_mat = topk_mask * iou_mat
        best_indices = masked_iou_mat.argmax(1)[:, None, :]
        best_target_mask = torch.zeros_like(duplicates, dtype=torch.bool)
        best_target_mask.scatter_(1, index=best_indices, src=~best_target_mask)
        topk_mask = torch.where(duplicates, best_target_mask, topk_mask)
        unique_indices = topk_mask.to(torch.uint8).argmax(dim=1)
        return unique_indices[..., None], topk_mask.any(dim=1), topk_mask

    def __call__(self, target: Tensor, predict: Tuple[Tensor, Tensor], return_indices: bool = False) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        predict_cls, predict_bbox = predict
        n_targets = target.shape[1]
        if n_targets == 0:
            device = predict_bbox.device
            align_cls = torch.zeros_like(predict_cls, device=device)
            align_bbox = torch.zeros_like(predict_bbox, device=device)
            valid_mask = torch.zeros(predict_cls.shape[:2], dtype=bool, device=device)
            anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
            if return_indices:
                matched_indices = torch.zeros(predict_cls.shape[:2], dtype=torch.long, device=device)
                return anchor_matched_targets, valid_mask, matched_indices
            return anchor_matched_targets, valid_mask

        target_cls, target_bbox = target.split([1, 4], dim=-1)
        target_cls = target_cls.long().clamp(0)

        grid_mask = self.get_valid_matrix(target_bbox)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        # STAL: Small-Target-Aware Label Assignment
        w = target_bbox[..., 2] - target_bbox[..., 0]
        h = target_bbox[..., 3] - target_bbox[..., 1]
        area = w * h
        # Small targets get a boost factor, prioritizing their selection
        stal_factor = 1.0 + 2.0 * torch.exp(-area / 1600.0) # [B, targets]

        target_matrix = (iou_mat**self.iou_factor) * (cls_mat**self.cls_factor) * stal_factor[:, :, None]

        topk_targets, topk_mask = self.filter_topk(target_matrix, grid_mask, topk=self.topk)
        topk_mask = self.ensure_one_anchor(target_matrix, topk_mask)
        unique_indices, valid_mask, topk_mask = self.filter_duplicates(iou_mat, topk_mask)

        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls_indices = torch.gather(target_cls, 1, unique_indices)
        align_cls = torch.zeros_like(align_cls_indices, dtype=torch.bool).repeat(1, 1, self.num_classes)
        align_cls.scatter_(-1, index=align_cls_indices, src=~align_cls)

        iou_mat *= topk_mask
        target_matrix *= topk_mask
        max_target = target_matrix.amax(dim=-1, keepdim=True)
        max_iou = iou_mat.amax(dim=-1, keepdim=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
        if return_indices:
            return anchor_matched_targets, valid_mask, unique_indices.squeeze(-1)
        return anchor_matched_targets, valid_mask


class YOLO26Vec2Box:
    def __init__(self, strides: List[int], image_size: List[int], num_classes: int, device: torch.device):
        self.strides = strides
        self.num_classes = num_classes
        self.device = device
        anchor_grid, scaler = generate_anchors(image_size, strides)
        self.image_size = image_size
        self.anchor_grid = anchor_grid.to(device)
        self.scaler = scaler.to(device)

    def update(self, image_size: List[int]):
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid = anchor_grid.to(self.device)
        self.scaler = scaler.to(self.device)

    def __call__(self, predicts: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            predicts: List of [P3, P4, P5] tensors from detection head
                     Each (B, 4 + nc, H, W)
        """
        preds_cls_list = []
        preds_box_list = []

        for pred in predicts:
            B, C, H, W = pred.shape
            pred_box_raw = pred[:, :4, :, :]
            pred_cls = pred[:, 4:, :, :]

            pred_cls = pred_cls.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            preds_cls_list.append(pred_cls)

            pred_box = pred_box_raw.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            preds_box_list.append(pred_box)

        preds_cls = torch.cat(preds_cls_list, dim=1)
        preds_box = torch.cat(preds_box_list, dim=1)

        # Convert LTRB distances to xyxy coordinates (pixel space)
        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)

        return preds_cls, preds_box


class YOLO26Loss:
    """YOLO26 loss function with ProgLoss and STAL."""

    def __init__(
        self,
        num_classes: int,
        strides: List[int],
        image_size: Optional[List[int]],
        device: torch.device,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        topk: int = 10,
        iou_factor: float = 6.0,
        cls_factor: float = 0.5,
    ):
        self.num_classes = num_classes
        self.strides = strides
        self.device = device
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.topk = topk
        self.iou_factor = iou_factor
        self.cls_factor = cls_factor

        self.cls_loss = BCELoss()
        self.box_loss = BoxLoss()

        self.matcher = None
        self.vec2box = None

        if image_size is not None:
            self._init_vec2box(image_size)

    def _init_vec2box(self, image_size: List[int]):
        self.vec2box = YOLO26Vec2Box(
            strides=self.strides,
            image_size=image_size,
            num_classes=self.num_classes,
            device=self.device,
        )
        self.matcher = YOLO26BoxMatcher(
            num_classes=self.num_classes,
            anchor_grid=self.vec2box.anchor_grid,
            scaler=self.vec2box.scaler,
            topk=self.topk,
            iou_factor=self.iou_factor,
            cls_factor=self.cls_factor,
        )

    def update_anchors(self, image_size: List[int]):
        if self.vec2box is None or self.vec2box.image_size != image_size:
            self._init_vec2box(image_size)

    def __call__(
        self, predictions: List[Tensor], targets: Tensor, epoch: Optional[int] = None, max_epochs: Optional[int] = None
    ) -> Dict[str, Tensor]:
        if self.vec2box is None:
            raise RuntimeError("Vec2Box not initialized. Call update_anchors() first.")

        preds_cls, preds_box = self.vec2box(predictions)

        B = targets.shape[0]
        W, H = self.vec2box.image_size
        scale = torch.tensor([1, W, H, W, H], device=targets.device, dtype=targets.dtype)
        targets_scaled = targets * scale

        align_targets, valid_masks = self.matcher(
            targets_scaled, (preds_cls.detach(), preds_box.detach())
        )

        targets_cls, targets_bbox = torch.split(align_targets, (self.num_classes, 4), dim=-1)

        preds_box_norm = preds_box / self.vec2box.scaler[None, :, None]
        targets_bbox_norm = targets_bbox / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        loss_cls = self.cls_loss(preds_cls, targets_cls, cls_norm)
        if valid_masks.any():
            loss_box = self.box_loss(
                preds_box_norm, targets_bbox_norm, valid_masks, box_norm, cls_norm
            )
        else:
            loss_box = preds_box_norm.sum() * 0.0

        # ProgLoss: Progressive Loss Balancing
        if epoch is not None and max_epochs is not None and max_epochs > 0:
            progress = min(1.0, max(0.0, float(epoch) / float(max_epochs)))
        else:
            progress = 1.0

        box_scale = 1.2 - 0.4 * progress
        cls_scale = 0.8 + 0.4 * progress

        loss_box_weighted = (self.box_weight * box_scale) * loss_box
        loss_cls_weighted = (self.cls_weight * cls_scale) * loss_cls

        total_loss = loss_box_weighted + loss_cls_weighted

        return {
            "total_loss": total_loss,
            "box_loss": loss_box_weighted,
            "cls_loss": loss_cls_weighted,
            "dfl_loss": torch.zeros_like(total_loss),
            "box": loss_box_weighted.item() if isinstance(loss_box_weighted, Tensor) else loss_box_weighted,
            "cls": loss_cls_weighted.item() if isinstance(loss_cls_weighted, Tensor) else loss_cls_weighted,
            "dfl": 0.0,
            "num_fg": valid_masks.sum().item() / max(B, 1),
        }


class YOLO26E2ELoss:
    """Combined dual-branch loss for YOLO26 end-to-end training."""

    def __init__(
        self,
        num_classes: int,
        strides: List[int],
        image_size: Optional[List[int]],
        device: torch.device,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        topk_many: int = 10,
        topk_one: int = 1,
        iou_factor: float = 6.0,
        cls_factor: float = 0.5,
    ):
        self.dense_loss = YOLO26Loss(
            num_classes=num_classes,
            strides=strides,
            image_size=image_size,
            device=device,
            box_weight=box_weight,
            cls_weight=cls_weight,
            topk=topk_many,
            iou_factor=iou_factor,
            cls_factor=cls_factor,
        )
        self.exclusive_loss = YOLO26Loss(
            num_classes=num_classes,
            strides=strides,
            image_size=image_size,
            device=device,
            box_weight=box_weight,
            cls_weight=cls_weight,
            topk=topk_one,
            iou_factor=iou_factor,
            cls_factor=cls_factor,
        )

    def update_anchors(self, image_size: List[int]):
        self.dense_loss.update_anchors(image_size)
        self.exclusive_loss.update_anchors(image_size)

    def __call__(
        self, dense_preds, exclusive_preds, targets, epoch: Optional[int] = None, max_epochs: Optional[int] = None
    ) -> Dict[str, Tensor]:
        loss_many = self.dense_loss(dense_preds, targets, epoch=epoch, max_epochs=max_epochs)
        loss_one = self.exclusive_loss(exclusive_preds, targets, epoch=epoch, max_epochs=max_epochs)

        total_loss = loss_many["total_loss"] + loss_one["total_loss"]
        box_loss = loss_many["box_loss"] + loss_one["box_loss"]
        cls_loss = loss_many["cls_loss"] + loss_one["cls_loss"]

        num_fg = loss_many.get("num_fg", 0) + loss_one.get("num_fg", 0)
        if isinstance(num_fg, Tensor):
            num_fg = num_fg.item()

        return {
            "total_loss": total_loss,
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "dfl_loss": torch.zeros_like(total_loss),
            "box": box_loss.item() if isinstance(box_loss, Tensor) else box_loss,
            "cls": cls_loss.item() if isinstance(cls_loss, Tensor) else cls_loss,
            "dfl": 0.0,
            "num_fg": num_fg,
        }
