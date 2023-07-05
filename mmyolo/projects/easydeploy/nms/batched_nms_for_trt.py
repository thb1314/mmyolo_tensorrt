# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from .onnx_helper import add_dummy_nms_for_onnx

_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                           [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                          dtype=torch.float32)




def batched_nms_for_trt(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_boxes_per_class: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    if box_coding == 1:
        boxes = boxes @ (_XYWH2XYXY.to(boxes.device))

    batched_dets, batched_scores, batched_labels = add_dummy_nms_for_onnx(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, pre_top_k, keep_top_k, labels=None)
    num_dets = (batched_scores > 0).sum(-1, keepdim=False)


    return num_dets, batched_dets, batched_scores, batched_labels.to(torch.int32)
