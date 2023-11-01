# Copyright (c) OpenMMLab. All rights reserved.
from projects.easydeploy.model import ORTWrapper, TRTWrapper  # isort:skip
import os
import random
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.utils import ProgressBar, path

from mmyolo.utils import register_all_modules
from mmyolo.utils.misc import get_file_list
import json
import os
import glob
import tensorrt as trt


logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
cur_dir = os.path.dirname(os.path.realpath(__file__))
so_files = glob.glob(os.path.join(cur_dir, '**', '*.so'), recursive=True)
import ctypes

for so_file in so_files:
    ctypes.cdll.LoadLibrary(so_file)
    print('load {} success!'.format(os.path.basename(so_file)))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('torch_checkpoint', help='Pytorch Checkpoint file')
    parser.add_argument(
        '--out-dir', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--result-json-path', default=None, help='Path to save the inference result')
    args = parser.parse_args()
    return args


def preprocess(config):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    bgr2rgb = data_preprocess.get('bgr_to_rgb', False)
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x[None].float()
            if bgr2rgb:
                x = x[:, [2,1,0], ...]
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x

    return PreProcess().eval()


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]

    # build the model from a config file and a checkpoint file
    if args.checkpoint.endswith('.onnx'):
        model = ORTWrapper(args.checkpoint, args.device)
    elif args.checkpoint.endswith('.engine') or args.checkpoint.endswith(
            '.plan'):
        model = TRTWrapper(args.checkpoint, args.device)
    else:
        raise NotImplementedError

    model.to(args.device)

    cfg = Config.fromfile(args.config)
    class_names = cfg.get('class_name')

    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline)

    pre_pipeline = preprocess(cfg)

    if not args.show and args.out_dir:
        path.mkdir_or_exist(args.out_dir)

    # build torch model
    torch_config = Config.fromfile(args.config)
    if 'init_cfg' in torch_config.model.backbone:
        torch_config.model.backbone.init_cfg = None
    torch_model = init_detector(torch_config, args.torch_checkpoint, device=args.device, cfg_options={})


    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))

    json_results = dict()

    for i, file in enumerate(files):
        # torch inference
        torch_result = inference_detector(torch_model, file)
        torch_labels = torch_result.pred_instances.labels
        torch_scores = torch_result.pred_instances.scores
        torch_bboxes = torch_result.pred_instances.bboxes
        print('torch_labels, torch_scores, torch_bboxes')
        print(torch_labels, torch_scores, torch_bboxes)
        # other framework inference
        bgr = mmcv.imread(file, channel_order='bgr')
        # rgb = mmcv.imconvert(bgr, 'bgr', 'rgb')
        data, samples = test_pipeline(dict(img=bgr, img_id=i)).values()
        pad_param = samples.get('pad_param',
                                np.array([0, 0, 0, 0], dtype=np.float32))
        h, w = samples.get('ori_shape', bgr.shape[:2])
        pad_param = torch.asarray(
            [pad_param[2], pad_param[0], pad_param[2], pad_param[0]],
            device=args.device)
        scale_factor = samples.get('scale_factor', [1., 1.])
        scale_factor = torch.asarray(scale_factor * 2, device=args.device)
        data = pre_pipeline(data).to(args.device)
        print('data.shape', data.shape)
        result = model(data)
        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show or args.out_dir is None else os.path.join(args.out_dir, filename)

        # Get candidate predict info by num_dets
        ori_scores = None
        ori_boxes = None
        topk_values = None
        topk_indices = None
        if len(result) >= 6 and getattr(result, 'name_496', None) is not None and getattr(result, 'onnx__Shape_497', None) is not None:
            topk_values = result.name_496
            topk_indices = result.onnx__Shape_497
            num_dets = result.num_dets
            bboxes = result.boxes
            scores = result.scores
            labels = result.labels
        elif len(result) >= 6 and 'scores.1' in result and 'TRT::EfficientNMS_TRT_472' in result:
            ori_scores = result['scores.1']
            ori_boxes = result['TRT::EfficientNMS_TRT_472']
            num_dets = result['num_dets']
            bboxes = result['boxes']
            scores = result['scores']
            labels = result['labels']
        else:
            assert len(result) == 4
            num_dets, bboxes, scores, labels = result
            if num_dets.shape[-1] != 1:
                scores, bboxes, num_dets, labels = result

        # print('scores, bboxes, num_dets, labels')
        # print(scores, bboxes, num_dets, labels)

        if hasattr(torch_model, 'bbox_head') and hasattr(torch_model.bbox_head, 'flatten_info'):
            torch_flatten_info = torch_model.bbox_head.flatten_info
            flatten_scores = torch_flatten_info.scores
            flatten_bboxes = torch_flatten_info.bboxes
            if ori_scores is not None and ori_boxes is not None:
                scores_diff = (ori_scores - flatten_scores).abs()
                bboxes_diff = (ori_boxes - flatten_bboxes).abs()
                if scores_diff.max() > 1e-2 or bboxes_diff.max() > 1:
                    print('\n找到了1')
                    print(scores_diff.max(), scores_diff.mean(), scores_diff.sum())
                    print(bboxes_diff.max(), bboxes_diff.mean(), bboxes_diff.sum())

        if hasattr(torch_model, 'bbox_head') and hasattr(torch_model.bbox_head, 'topk_infos'):
            torch_topk_info = torch_model.bbox_head.topk_infos[0]
            torch_topk_values = torch_topk_info.topk_values
            torch_topk_indices = torch_topk_info.topk_indices
            if topk_values is not None and topk_indices is not None:
                values_diff = (torch_topk_values - topk_values).abs().float()
                indices_diff = (torch_topk_indices - topk_indices).abs().float()
                for item in indices_diff.view(-1, 2):
                    print(item)
                if values_diff.max() > 1e-2 or indices_diff.max() > 1:
                    print('\n找到了2')
                    print(values_diff.max(), values_diff.mean(), values_diff.sum())
                    print(indices_diff.max(), indices_diff.mean(), indices_diff.sum())

        num_dets = min(torch_scores.shape[-1], num_dets.item())

        torch_scores = torch_scores[:num_dets]
        torch_bboxes = torch_bboxes[:num_dets]
        torch_labels = torch_labels[:num_dets]

        scores = scores[0, :num_dets]
        bboxes = bboxes[0, :num_dets]
        labels = labels[0, :num_dets]
        if bboxes.shape[-1] == 4:
            bboxes -= pad_param
            bboxes /= scale_factor
            bboxes[:, 0::2].clamp_(0, w)
            bboxes[:, 1::2].clamp_(0, h)
        elif bboxes.shape[-1] == 5:
            from mmrotate.structures.bbox import RotatedBoxes
            from mmdet.structures.bbox.transforms import scale_boxes
            bboxes = RotatedBoxes(bboxes)
            if pad_param is not None:
                bboxes.translate_([-pad_param[2], -pad_param[0]])
            scale_factor = [1 / s for s in samples.get('scale_factor', [1., 1.])]
            bboxes = scale_boxes(bboxes, scale_factor)
            bboxes = bboxes.tensor
        
        print('scores, bboxes, num_dets, labels')
        print(scores, bboxes, num_dets, labels)
        json_results[file] = {
            'scores':scores.detach().cpu().numpy().tolist(),
            'bboxes':bboxes.detach().cpu().numpy().tolist(),
            'labels':labels.detach().cpu().numpy().tolist()
        }

        # compare torch result with other framework result
        scores_diff = (torch_scores - scores).abs()
        bboxes_diff = (torch_bboxes - bboxes).abs()
        labels_diff = (torch_labels - labels).float().abs()
        if num_dets > 0 and scores_diff.max() > 1e-2:
            print('\nbingo')
            print(scores_diff.max(), scores_diff.mean(), scores_diff.sum())
            print(bboxes_diff.max(), bboxes_diff.mean(), bboxes_diff.sum())
            print(labels_diff.max(), labels_diff.mean(), labels_diff.sum())

        progress_bar.update()

        if not args.show and out_file is None:
            continue

        bboxes = bboxes.round().int()
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.tolist()
            color = colors[label]

            if class_names is not None:
                label_name = class_names[label]
                name = f'cls:{label_name}_score:{score:0.4f}'
            else:
                name = f'cls:{label}_score:{score:0.4f}'

            cv2.rectangle(bgr, bbox[:2], bbox[2:], color, 2)
            cv2.putText(
                bgr,
                name, (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, [225, 255, 255],
                thickness=3)

        if args.show:
            mmcv.imshow(bgr, 'result', 0)
        elif out_file:
            mmcv.imwrite(bgr, out_file)
    
    if args.result_json_path is not None:
        os.makedirs(os.path.dirname(args.result_json_path), exist_ok=True)
        with open(args.result_json_path, 'w', encoding='utf-8') as fp:
            json.dump(json_results, fp, ensure_ascii=False)
        

if __name__ == '__main__':
    main()
