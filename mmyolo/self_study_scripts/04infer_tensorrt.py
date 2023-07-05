# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
from mmengine.config import Config, ConfigDict
from mmengine.utils import ProgressBar, path
from mmyolo.utils.misc import get_file_list
import mmyolo
from mmcv.transforms import Compose
from mmdet.utils.misc import get_test_pipeline_cfg
import numpy as np
from mmdet.evaluation import get_classes
import torch
from mmyolo.utils import register_all_modules
from self_study_scripts.trt_runner import TRTRunner
from argparse import ArgumentParser


def preprocess(config):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x[None].float()
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x

    return PreProcess().eval()

color_list = (
    (216 , 82 , 24),
    (236 ,176 , 31),
    (125 , 46 ,141),
    (118 ,171 , 47),
    ( 76 ,189 ,237),
    (238 , 19 , 46),
    ( 76 , 76 , 76),
    (153 ,153 ,153),
    (255 ,  0 ,  0),
    (255 ,127 ,  0),
    (190 ,190 ,  0),
    (  0 ,255 ,  0),
    (  0 ,  0 ,255),
    (170 ,  0 ,255),
    ( 84 , 84 ,  0),
    ( 84 ,170 ,  0),
    ( 84 ,255 ,  0),
    (170 , 84 ,  0),
    (170 ,170 ,  0),
    (170 ,255 ,  0),
    (255 , 84 ,  0),
    (255 ,170 ,  0),
    (255 ,255 ,  0),
    (  0 , 84 ,127),
    (  0 ,170 ,127),
    (  0 ,255 ,127),
    ( 84 ,  0 ,127),
    ( 84 , 84 ,127),
    ( 84 ,170 ,127),
    ( 84 ,255 ,127),
    (170 ,  0 ,127),
    (170 , 84 ,127),
    (170 ,170 ,127),
    (170 ,255 ,127),
    (255 ,  0 ,127),
    (255 , 84 ,127),
    (255 ,170 ,127),
    (255 ,255 ,127),
    (  0 , 84 ,255),
    (  0 ,170 ,255),
    (  0 ,255 ,255),
    ( 84 ,  0 ,255),
    ( 84 , 84 ,255),
    ( 84 ,170 ,255),
    ( 84 ,255 ,255),
    (170 ,  0 ,255),
    (170 , 84 ,255),
    (170 ,170 ,255),
    (170 ,255 ,255),
    (255 ,  0 ,255),
    (255 , 84 ,255),
    (255 ,170 ,255),
    ( 42 ,  0 ,  0),
    ( 84 ,  0 ,  0),
    (127 ,  0 ,  0),
    (170 ,  0 ,  0),
    (212 ,  0 ,  0),
    (255 ,  0 ,  0),
    (  0 , 42 ,  0),
    (  0 , 84 ,  0),
    (  0 ,127 ,  0),
    (  0 ,170 ,  0),
    (  0 ,212 ,  0),
    (  0 ,255 ,  0),
    (  0 ,  0 , 42),
    (  0 ,  0 , 84),
    (  0 ,  0 ,127),
    (  0 ,  0 ,170),
    (  0 ,  0 ,212),
    (  0 ,  0 ,255),
    (  0 ,  0 ,  0),
    ( 36 , 36 , 36),
    ( 72 , 72 , 72),
    (109 ,109 ,109),
    (145 ,145 ,145),
    (182 ,182 ,182),
    (218 ,218 ,218),
    (  0 ,113 ,188),
    ( 80 ,182 ,188),
    (127 ,127 ,  0),
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    args = parser.parse_args()
    return args


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()


    # build the model from a config file and a checkpoint file
    assert args.checkpoint.endswith('.onnx')
    trt_runner = TRTRunner(args.checkpoint)
    
    cfg = Config.fromfile(args.config)
    class_names = cfg.get('class_name')
    if class_names is None:
        class_names = get_classes('coco')

    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline)
    pre_pipeline = preprocess(cfg)

    path.mkdir_or_exist(args.out_dir)
    # get file list
    files, source_type = get_file_list(args.img)
    # start detector inference
    progress_bar = ProgressBar(len(files))
    image_det_result = dict()
    for i, file in enumerate(files):
        bgr = mmcv.imread(file)
        rgb = mmcv.imconvert(bgr, 'bgr', 'rgb')
        data, samples = test_pipeline(dict(img=rgb, img_id=i)).values()
        pad_param = samples.get('pad_param',
                                np.array([0, 0, 0, 0], dtype=np.float32))
        h, w = samples.get('ori_shape', rgb.shape[:2])
        pad_param = torch.asarray([pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
        scale_factor = samples.get('scale_factor', [1., 1])
        scale_factor = torch.asarray(scale_factor + scale_factor)
        data = pre_pipeline(data).cpu()

        result = trt_runner(data.numpy())

        ori_scores = result['scores.1']
        ori_boxes = result['TRT::EfficientNMS_TRT_472']
        flatten_cls_scores = torch.load(os.path.join(SCRIPT_DIR, 'flatten_cls_scores.pkl'), map_location='cpu').numpy()
        flatten_decoded_bboxes = torch.load(os.path.join(SCRIPT_DIR, 'flatten_decoded_bboxes.pkl'), map_location='cpu').numpy()
        print('flatten_cls_scores', flatten_cls_scores.shape)
        print('flatten_decoded_bboxes', flatten_decoded_bboxes.shape)
        scores_diff = np.abs(ori_scores - flatten_cls_scores)
        boxes_diff = np.abs(ori_boxes - flatten_decoded_bboxes)
        print(np.max(scores_diff))
        print(np.max(boxes_diff))

        # Get candidate predict info by num_dets
        num_dets, bboxes, scores, labels = result['num_dets'], result['boxes'], result['scores'], result['labels']
        num_dets = int(num_dets)
        scores = scores[0, :num_dets]
        bboxes = bboxes[0, :num_dets]
        labels = labels[0, :num_dets]

        bboxes -= pad_param.numpy()
        bboxes /= scale_factor.numpy()

        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
        # bboxes = np.round(bboxes).astype(np.int32)

        progress_bar.update()


if __name__ == '__main__':
    main()
