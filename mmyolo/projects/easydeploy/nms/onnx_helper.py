# Copyright (c) OpenMMLab. All rights reserved.
import os
from functools import reduce
import torch


def dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape):
    """Clip boxes dynamically for onnx.

    Since torch.clamp cannot have dynamic `min` and `max`, we scale the
      boxes by 1/max_shape and clamp in the range [0, 1].

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor or torch.Size): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert isinstance(
        max_shape,
        torch.Tensor), '`max_shape` should be tensor of (h,w) for onnx'

    # scale by 1/max_shape
    x1 = x1 / max_shape[1]
    y1 = y1 / max_shape[0]
    x2 = x2 / max_shape[1]
    y2 = y2 / max_shape[0]

    # clamp [0, 1]
    x1 = torch.clamp(x1, 0, 1)
    y1 = torch.clamp(y1, 0, 1)
    x2 = torch.clamp(x2, 0, 1)
    y2 = torch.clamp(y2, 0, 1)

    # scale back
    x1 = x1 * max_shape[1]
    y1 = y1 * max_shape[0]
    x2 = x2 * max_shape[1]
    y2 = y2 * max_shape[0]
    return x1, y1, x2, y2


def get_k_for_topk(k, size):
    """Get k of TopK for onnx exporting.

    The K of TopK in TensorRT should not be a Tensor, while in ONNX Runtime
      it could be a Tensor.Due to dynamic shape feature, we have to decide
      whether to do TopK and what K it should be while exporting to ONNX.
    If returned K is less than zero, it means we do not have to do
      TopK operation.

    Args:
        k (int or Tensor): The set k value for nms from config file.
        size (Tensor or torch.Size): The number of elements of \
            TopK's input tensor
    Returns:
        tuple: (int or Tensor): The final K for TopK.
    """
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if torch.onnx.is_in_onnx_export():
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if is_trt_backend:
            # TensorRT does not support dynamic K with TopK op
            if 0 < k < size:
                ret_k = k
        else:
            # Always keep topk op for dynamic input in onnx for ONNX Runtime
            ret_k = torch.where(k < size, k, size)
    elif k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k


def add_dummy_nms_for_onnx(boxes,
                           scores,
                           max_output_boxes_per_class=1000,
                           iou_threshold=0.5,
                           score_threshold=0.05,
                           pre_top_k=-1,
                           after_top_k=-1,
                           labels=None,
                           idxs=None):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes]
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (bool): Number of top K boxes to keep before nms.
            Defaults to -1.
        after_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        labels (Tensor, optional): It not None, explicit labels would be used.
            Otherwise, labels would be automatically generated using
            num_classed. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
            and class labels of shape [N, num_det].
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    
    batch_size = scores.shape[0]
    num_class = int(scores.shape[2])
    num_box = int(boxes.shape[1])

    nms_pre = torch.tensor(pre_top_k, device=scores.device, dtype=torch.long)
    nms_pre = get_k_for_topk(nms_pre, int(num_box * num_class))

    # scores = scores * (scores > score_threshold).float()
    if labels is None:
        labels = torch.arange(num_class, dtype=torch.long).to(scores.device)
        labels = labels.view(1, 1, num_class).repeat((batch_size, num_box, 1))

    boxes_for_nms = boxes
    if nms_pre > 0 and nms_pre != int(boxes.shape[1]):
        # topk_inds: [B, nms_pre]
        _, topk_inds = scores.view(-1, num_box * num_class).topk(nms_pre, dim=-1)
        batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds).long()
        topk_inds_for_box = topk_inds // num_class
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        transformed_inds_for_score = num_box * num_class * batch_inds + topk_inds
        transformed_inds_for_boxes = num_box * batch_inds + topk_inds_for_box
        topk_boxes = boxes.reshape(-1, 4)[transformed_inds_for_boxes, :].reshape(-1, nms_pre , 4)
        scores = scores.reshape(-1, 1)[transformed_inds_for_score, :].reshape(-1, nms_pre, 1)
        labels = labels.reshape(-1, 1)[transformed_inds_for_score, :].reshape(-1, nms_pre)
        
        # box shifting for different class
        # [b,1,1]
        # max_coordinate, _ = topk_boxes.max(dim=1, keepdim=True)
        # max_coordinate, _ = max_coordinate.max(dim=2, keepdim=True)
        max_coordinate = topk_boxes.max()
        # [b,N,1]
        offsets = labels * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = topk_boxes + offsets.view(1, nms_pre, 1)
        boxes = topk_boxes

    ori_num_class = num_class
    num_box = int(boxes.shape[1])
    num_class = int(scores.shape[2])
    scores = scores.permute(0, 2, 1)
    # turn off tracing to create a dummy output of nms
    state = torch._C._get_tracing_state()
    # dummy indices of nms's output
    num_fake_det = 2
    batch_inds = torch.randint(batch_size, (num_fake_det, 1))
    cls_inds = torch.randint(num_class, (num_fake_det, 1))
    box_inds = torch.randint(num_box, (num_fake_det, 1))
    indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
    output = indices
    setattr(DummyONNXNMSop, 'output', output)

    # open tracing
    torch._C._set_tracing_state(state)

    # box shifting for fpn level
    if idxs is not None:
        # [b,1,1]
        max_coordinate, _ = boxes_for_nms.max(dim=1, keepdim=True)
        max_coordinate, _ = max_coordinate.max(dim=2, keepdim=True)
        # [b,N,1]
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes_for_nms + offsets.view(1, num_box, 1)

    selected_indices = DummyONNXNMSop.apply(boxes_for_nms, scores,
                                            max_output_boxes_per_class * ori_num_class // num_class,
                                            iou_threshold, score_threshold)

    batch_inds, cls_inds = selected_indices[:, 0], selected_indices[:, 1]
    box_inds = selected_indices[:, 2]
    scores = scores.reshape(-1, 1)
    if num_class != 1:
        boxes = boxes.reshape(-1, 1, num_box, 4).repeat(1, num_class, 1, 1).reshape(-1, 4)
    pos_inds = (num_class * batch_inds + cls_inds) * num_box + box_inds
    mask = scores.new_zeros(scores.shape)
    # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
    # PyTorch style code: mask[batch_inds, box_inds] += 1
    mask[pos_inds, :] += 1
    scores = scores * mask
    boxes = boxes * mask

    scores_last_shapes = list(map(int, scores.shape))
    scores_last_dims = reduce(lambda x, y: x * y, scores_last_shapes, 1)
    scores = scores.reshape(-1, scores_last_dims // int(batch_size))

    boxes_last_shapes = list(map(int, boxes.shape))
    boxes_last_dims = reduce(lambda x, y: x * y, boxes_last_shapes, 1)
    boxes = boxes.reshape(-1, boxes_last_dims // 4 // int(batch_size), 4)

    labels_last_shapes = list(map(int, labels.shape))
    labels_last_dims = reduce(lambda x, y: x * y, labels_last_shapes, 1)
    labels = labels.reshape(-1, labels_last_dims // int(batch_size))

    nms_after = torch.tensor(after_top_k, device=scores.device, dtype=torch.long)
    nms_after = get_k_for_topk(nms_after, num_box * num_class)

    if nms_after > 0:
        _, topk_inds = scores.topk(nms_after)
        if int(batch_size) == 1:
            transformed_inds = topk_inds
        else:
            batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
            # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
            transformed_inds = int(scores.shape[1]) * batch_inds + topk_inds
        scores = scores.reshape(-1, 1)[transformed_inds, :]
        scores_last_shapes = list(map(int, scores.shape))
        scores_last_dims = reduce(lambda x, y: x * y, scores_last_shapes, 1)
        scores = scores.reshape(-1, scores_last_dims // int(batch_size))

        boxes = boxes.reshape(-1, 4)[transformed_inds, :]
        boxes_last_shapes = list(map(int, boxes.shape))
        boxes_last_dims = reduce(lambda x, y: x * y, boxes_last_shapes, 1)
        boxes = boxes.reshape(-1, boxes_last_dims // 4 // int(batch_size), 4)
        
        
        labels = labels.reshape(-1, 1)[transformed_inds, :]
        labels_last_shapes = list(map(int, labels.shape))
        labels_last_dims = reduce(lambda x, y: x * y, labels_last_shapes, 1)
        labels = labels.reshape(-1, labels_last_dims // int(batch_size))

    return boxes, scores, labels



class DummyONNXNMSop(torch.autograd.Function):
    """DummyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):

        return DummyONNXNMSop.output

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold)
