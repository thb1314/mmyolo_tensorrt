# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
from mmengine.config import Config, ConfigDict
from mmengine.utils import ProgressBar, path
from mmyolo.utils.misc import get_file_list
import logging
from mmcv.transforms import Compose
from mmdet.utils.misc import get_test_pipeline_cfg
import numpy as np
from mmdet.evaluation import get_classes
import torch
from mmyolo.utils import register_all_modules
from self_study_scripts.trt_runner import TRTRunner
from argparse import ArgumentParser
import json
import random
import glob
import tensorrt as trt
import pycuda.driver as cuda
import tqdm
from io import BytesIO


from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
import mqbench
from mqbench.advanced_ptq import ptq_reconstruction

from mmdet.apis import inference_detector, init_detector


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


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
    parser.add_argument('onnx_file', help='onnx file')
    parser.add_argument(
        '--max-batch-size', default=1, type=int, help='Path to output file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--enable-int8', action='store_true', default=False, help='Whether enable tensorrt int8 inference')
    parser.add_argument(
        '--dynamic-range-file', type=str, default=None)
    args = parser.parse_args()
    return args


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    assert args.onnx_file.endswith('.onnx')
    max_batch_size = args.max_batch_size
        
    cfg = Config.fromfile(args.config)
    class_names = cfg.get('class_name')
    if class_names is None:
        class_names = get_classes('coco')

    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline)
    pre_pipeline = preprocess(cfg)

    path.mkdir_or_exist(args.out_dir)

    onnx_filepath = args.onnx_file
    
    import onnx
    def change_input_dim(model):
        sym_batch_dim = "N"
        inputs = model.graph.input
        for input in inputs:
            dim1 = input.type.tensor_type.shape.dim[0]
            dim1.dim_param = sym_batch_dim                        
    def apply(transform, infile, outfile):
        model = onnx.load(infile)
        transform(model)
        onnx.save(model, outfile)
    apply(change_input_dim, onnx_filepath, onnx_filepath)

    dynamic_range_file = args.dynamic_range_file
    trt_runner = TRTRunner(onnx_filepath, max_batch_size=max_batch_size, enable_fp16=True, int8_calibrator=None,
                           dynamic_range_file=dynamic_range_file)
    # get file list
    files, _ = get_file_list(args.img)
    # start detector inference
    len_files = len(files)
    image_det_result = dict()
    progress_bar = ProgressBar((len_files + max_batch_size - 1) // max_batch_size)
    for start_index in range(0, len_files, max_batch_size):
        end_index = min(len_files, start_index + max_batch_size)
        data_list = []
        shape_list = []
        pad_param_list = []
        scale_factor_list = []
        file_list = []

        for i, file in enumerate(files[start_index:end_index]):
            bgr = mmcv.imread(file)
            rgb = mmcv.imconvert(bgr, 'bgr', 'rgb')
            data, samples = test_pipeline(dict(img=rgb, img_id=(start_index + i))).values()
            pad_param = samples.get('pad_param',
                                    np.array([0, 0, 0, 0], dtype=np.float32))
            h, w = samples.get('ori_shape', rgb.shape[:2])
            pad_param = torch.asarray([pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
            scale_factor = samples.get('scale_factor', [1., 1])
            scale_factor = torch.asarray(scale_factor + scale_factor)
            data = pre_pipeline(data).cpu()

            data_list.append(data)
            shape_list.append((h, w))
            pad_param_list.append(pad_param)
            scale_factor_list.append(scale_factor)
            file_list.append(file)

        data_list = np.concatenate([data.numpy() for data in data_list], axis=0)
        
        result = trt_runner(data_list)

        # Get candidate predict info by num_dets
        batched_num_dets, batched_bboxes, batched_scores, batched_labels = result['num_dets'], result['boxes'], result['scores'], result['labels']
        for num_dets, bboxes, scores, labels, pad_param, scale_factor, (h, w), file in \
            zip(batched_num_dets, batched_bboxes, batched_scores, batched_labels, 
                pad_param_list, scale_factor_list, shape_list, file_list):
            num_dets = int(num_dets)
            scores = scores[:num_dets]
            bboxes = bboxes[:num_dets]
            labels = labels[:num_dets]

            bboxes -= pad_param.numpy()
            bboxes /= scale_factor.numpy()

            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
            # bboxes = np.round(bboxes).astype(np.int32)

            image_det_result[file] = {
                'scores':scores.tolist(),
                'bboxes':bboxes.tolist(),
                'labels':labels.tolist()
            }
        progress_bar.update()

    with open(os.path.join(args.out_dir, 'result.json'), 'w', encoding='utf-8') as fp:
        json.dump(image_det_result, fp, ensure_ascii=False)


if __name__ == '__main__':
    main()
