import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Union

from ..yolo9.utils import preprocess_numpy, preprocess_image
from ...postprocess.yolo9 import (
    ImageSize,
    _input_size_hw,
    _process_masks,
    _xywhr_to_xyxy,
)

def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: ImageSize = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = True,
) -> Dict:
    """Postprocess YOLO26 outputs using top-K selection (no NMS)."""
    del iou_thres # not used

    input_h, input_w = _input_size_hw(input_size)
    predictions = output["predictions"] # (B, 4+nc, total_anchors) or (4+nc, total_anchors)
    is_obb = bool(output.get("obb", False))
    proto = output.get("proto")
    mask_coeffs = output.get("mask_coeffs")
    keypoints_all = output.get("keypoints")

    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)

    # Batch size is assumed 1 for post-processing in inference
    pred = predictions[0].transpose(0, 1) # (total_anchors, 4+nc)
    num_anchors = pred.shape[0]

    boxes_input = pred[:, :4]
    if is_obb:
        angles_input = pred[:, 4]
        scores = pred[:, 5:]
    else:
        angles_input = None
        scores = pred[:, 4:]

    num_classes = scores.shape[-1]
    topk_anchors = min(max_det, num_anchors)
    if topk_anchors == 0 or num_classes == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Stage 1: select top-K anchors by their best class score
    anchor_scores, class_ids = torch.max(scores, dim=-1)
    anchor_scores, anchor_indices = torch.topk(anchor_scores, topk_anchors, dim=-1)

    boxes = torch.gather(boxes_input, dim=0, index=anchor_indices.unsqueeze(-1).expand(-1, 4))
    scores = torch.gather(scores, dim=0, index=anchor_indices.unsqueeze(-1).expand(-1, num_classes))
    class_ids = torch.gather(class_ids, dim=0, index=anchor_indices)

    if angles_input is not None:
        angles = torch.gather(angles_input, dim=0, index=anchor_indices)
    else:
        angles = None

    if keypoints_all is not None:
        kpts = keypoints_all[0] if keypoints_all.dim() == 4 else keypoints_all
        keypoints = torch.gather(kpts, dim=0, index=anchor_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, kpts.shape[1], kpts.shape[2]))
    else:
        keypoints = None

    if mask_coeffs is not None:
        coeffs_all = mask_coeffs[0].transpose(0, 1) if mask_coeffs.dim() == 3 else mask_coeffs
        coeffs = torch.gather(coeffs_all, dim=0, index=anchor_indices.unsqueeze(-1).expand(-1, coeffs_all.shape[-1]))
    else:
        coeffs = None

    # Filter by confidence threshold
    keep = anchor_scores > conf_thres
    if not keep.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes = boxes[keep]
    scores_out = anchor_scores[keep]
    class_ids = class_ids[keep]
    if angles is not None:
        angles = angles[keep]
    if keypoints is not None:
        keypoints = keypoints[keep]
    if coeffs is not None:
        coeffs = coeffs[keep]

    # Convert coordinates to pixel space / original size
    boxes_orig = boxes.clone()
    if original_size is not None:
        if letterbox:
            orig_w, orig_h = original_size
            ratio = min(input_h / orig_h, input_w / orig_w)
            boxes_orig[:, :4] = boxes_orig[:, :4] / ratio
            if keypoints is not None:
                keypoints = keypoints.clone()
                keypoints[..., :2] = keypoints[..., :2] / ratio
        else:
            orig_w, orig_h = original_size
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            boxes_orig[:, [0, 2]] *= scale_x
            boxes_orig[:, [1, 3]] *= scale_y
            if keypoints is not None:
                keypoints = keypoints.clone()
                keypoints[..., 0] *= scale_x
                keypoints[..., 1] *= scale_y

        boxes_orig[:, [0, 2]] = torch.clamp(boxes_orig[:, [0, 2]], 0, original_size[0])
        boxes_orig[:, [1, 3]] = torch.clamp(boxes_orig[:, [1, 3]], 0, original_size[1])
        if keypoints is not None:
            keypoints[..., 0] = torch.clamp(keypoints[..., 0], 0, original_size[0])
            keypoints[..., 1] = torch.clamp(keypoints[..., 1], 0, original_size[1])

    # Validate box dimensions
    widths = boxes_orig[:, 2] - boxes_orig[:, 0]
    heights = boxes_orig[:, 3] - boxes_orig[:, 1]
    valid = (widths > 0) & (heights > 0)
    if not valid.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes_orig = boxes_orig[valid]
    scores_out = scores_out[valid]
    class_ids = class_ids[valid]
    if angles is not None:
        angles = angles[valid]
    if keypoints is not None:
        keypoints = keypoints[valid]
    if coeffs is not None:
        coeffs = coeffs[valid]
        boxes = boxes[valid] # keep un-rescaled boxes for mask cropping

    result = {
        "boxes": boxes_orig.detach().cpu().numpy().tolist(),
        "scores": scores_out.detach().cpu().numpy().tolist(),
        "classes": class_ids.detach().cpu().numpy().tolist(),
        "num_detections": len(boxes_orig),
    }

    if is_obb and angles is not None:
        wh = (boxes_orig[:, 2:4] - boxes_orig[:, 0:2]).clamp_min(0)
        centers = (boxes_orig[:, 0:2] + boxes_orig[:, 2:4]) / 2
        xywhr = torch.cat((centers, wh, angles[:, None]), dim=1)
        obb_out = torch.cat(
            (xywhr, scores_out[:, None], class_ids[:, None].float()),
            dim=1,
        )
        # OBB xyxy boxes
        boxes_obb = _xywhr_to_xyxy(xywhr)
        if original_size is not None:
            boxes_obb[:, [0, 2]] = torch.clamp(boxes_obb[:, [0, 2]], 0, original_size[0])
            boxes_obb[:, [1, 3]] = torch.clamp(boxes_obb[:, [1, 3]], 0, original_size[1])
        result["boxes"] = boxes_obb.detach().cpu().numpy().tolist()
        result["obb"] = obb_out.detach().cpu().numpy().tolist()

    if keypoints is not None:
        result["keypoints"] = keypoints.detach().cpu()

    if coeffs is not None and proto is not None:
        proto_i = proto[0] if proto.dim() == 4 else proto
        masks = _process_masks(
            proto_i,
            coeffs,
            boxes,
            input_shape=(input_h, input_w),
            original_size=original_size,
            letterbox=letterbox,
        )
        result["masks"] = masks.detach().cpu()

    return result
