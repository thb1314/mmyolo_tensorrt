from self_study_scripts.trt_runner import TRTRunner
from self_study_scripts.quant_utils import quant_model_init, QuantConfig, zero_scale_fix_
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
from argparse import ArgumentParser
import json
import random
import glob
import tensorrt as trt
import pycuda.driver as cuda
import tqdm
from io import BytesIO
from copy import deepcopy
from self_study_scripts.utils import de_parallel
from mmdet.apis import inference_detector, init_detector
from torch.utils.data import DataLoader, Dataset
import copy
from functools import partial
from pytorch_quantization import nn as quant_nn
from self_study_scripts.utils import module_quant_disable, module_quant_enable, get_module, model_quant_disable, model_quant_enable


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


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
                x = x[:, [2, 1, 0], ...]
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x

    return PreProcess().eval()


def sqnr(x: torch.Tensor, y: torch.Tensor):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return (20 * torch.log10(Ps / (Pn + 1e-5))).item()


def cosine(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """calulate the cosine similarity between x and y"""
    if x.shape != y.shape:
        raise ValueError(f'Can not compute loss for tensors with different shape. ({x.shape} and {y.shape})')
    reduction = str(reduction).lower()

    if x.ndim == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    x = x.flatten(start_dim=1).float()
    y = y.flatten(start_dim=1).float()

    cosine_sim = torch.cosine_similarity(x, y, dim=-1)

    if reduction == 'mean':
        return torch.mean(cosine_sim)
    elif reduction == 'sum':
        return torch.sum(cosine_sim)
    elif reduction == 'none':
        return cosine_sim
    else:
        raise ValueError(f'Cosine similarity do not supported {reduction} method.')


def eval_function_by_metric_function(det_model, dataloader, metric_function):

    device = next(det_model.parameters()).device
    len_files = len(dataloader)
    progress_bar = ProgressBar(len_files)

    metrics_history = {}

    names_dict = {}
    inputs_dict = {}
    outputs_dict = {}
    hooks = []
    
    
    # disable all quantable layer
    model_quant_disable(det_model)

    def forward_hook(module, input, output):
        name = names_dict[module]
        inputs_dict[name] = input
        outputs_dict[name] = output

    for k, m in det_model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
           isinstance(m, quant_nn.QuantConvTranspose2d) or \
           isinstance(m, quant_nn.MaxPool2d):
            names_dict[m] = k
            inputs_dict[k] = None
            # hooks.append(m.register_forward_hook(forward_hook))
    
    def calc_matric(xs, ys):
        metric_value = 0
        if isinstance(xs, (list, tuple)):
            for x, y in zip(xs, ys):
                metric_value += calc_matric(x, y)
        else:
                metric_value += metric_function(xs, ys)
        return metric_value

    for i, input_tensor in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            ori_outputs = det_model(input_tensor, mode="tensor")
            for key in inputs_dict:
                # input_value = inputs_dict[key]
                # output_value = outputs_dict[key]
                # if isinstance(output_value, tuple):
                #     output_value = output_value[0]
                # module = get_module(det_model, key)
                module_quant_enable(det_model, key)
                quantized_outputs = det_model(input_tensor, mode="tensor")
                #     quantized_value = module(*input_value)
                module_quant_disable(det_model, key)
                metric_value = calc_matric(ori_outputs, quantized_outputs)
                key_metrics_history = metrics_history.setdefault(key, [])
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.detach().cpu().item()
                key_metrics_history.append(metric_value)

        progress_bar.update()
        
    # disable all quantable layer
    model_quant_enable(det_model)

    # for hook in hooks:
    #     hook.remove()

    def reduce_mean(metrics):
        len_metrics = len(metrics)
        return sum(metrics) / len_metrics

    return [(key, reduce_mean(value)) for key, value in metrics_history.items()]



def eval_function_by_map(det_model, filepaths):
    len_files = len(filepaths)
    max_batch_size = 32
    image_torch_det_result = dict()
    progress_bar = ProgressBar((len_files + max_batch_size - 1) // max_batch_size)

    for start_index in range(0, len_files, max_batch_size):
        end_index = min(len_files, start_index + max_batch_size)

        sub_files = filepaths[start_index:end_index]
        torch_results = inference_detector(det_model, sub_files)
        for i, file in enumerate(sub_files):
            torch_result = torch_results[i]
            torch_labels = torch_result.pred_instances.labels
            torch_scores = torch_result.pred_instances.scores
            torch_bboxes = torch_result.pred_instances.bboxes
            image_torch_det_result[file] = {
                'scores':torch_scores.cpu().numpy().tolist(),
                'bboxes':torch_bboxes.cpu().numpy().tolist(),
                'labels':torch_labels.cpu().numpy().tolist()
            }
        progress_bar.update()

    import mmyolo
    mmyolo_dir = os.path.dirname(os.path.dirname(mmyolo.__file__))
    coco_root = os.path.join(mmyolo_dir, 'data', 'coco')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(os.path.join(coco_root, 'annotations', 'instances_val2017.json'))
    coco_basename2imgid = {img_info['file_name']:img_id for img_id, img_info in coco_gt.imgs.items()}
    coco_classes = [(v["id"], v["name"]) for k, v in coco_gt.cats.items()]
    annotations = []
    for filepath, info in image_torch_det_result.items():
        basename = os.path.basename(filepath)
        img_id = coco_basename2imgid[basename]
        scores = info['scores']
        bboxes = info['bboxes']
        labels = info['labels']

        for score, box, label in zip(scores, bboxes, labels):
            cat_id = coco_classes[int(label)][0]
            x1, y1, x2, y2 = box
            box_width = max(0, (x2 - x1))
            box_height = max(0, (y2 - y1))
            annotations.append({
                'bbox': [x1, y1, box_width, box_height],
                'category_id': cat_id,
                'image_id': img_id,
                'score': score
            })

    coco_pred = coco_gt.loadRes(annotations)
    coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # 返回map
    return coco_evaluator.stats[0]


def quant_sensitivity_analyse_by_map(model_ptq, eval_function):
    # disable all quantable layer
    model_quant_disable(model_ptq)
    
    # analyse each quantable layer
    quant_sensitivity = list()
    # eval_result = eval_function(model_ptq)
    # quant_sensitivity.append((None, eval_result))
    
    for k, m in model_ptq.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
           isinstance(m, quant_nn.QuantConvTranspose2d) or \
           isinstance(m, quant_nn.MaxPool2d):
            module_quant_enable(model_ptq, k)
        else:
            # module can not be quantized, continue
            continue

        eval_result = eval_function(model_ptq)
        print(eval_result)
        print("Quantize Layer {}, result {}".format(k, eval_result))
        quant_sensitivity.append((k, eval_result))
        # disable this module sensitivity, anlayse next module
        module_quant_disable(model_ptq, k)

    return quant_sensitivity

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('torch_checkpoint', help='Pytorch Checkpoint file')
    parser.add_argument(
        '--calib-imgs', type=str, help='Path to calibiration image filepath')
    parser.add_argument(
        '--max-batch-size', default=1, type=int, help='Path to output file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--enable-fp16', action='store_true', default=False, help='Whether enable tensorrt fp16 inference')
    parser.add_argument(
        '--enable-int8', action='store_true', default=False, help='Whether enable tensorrt int8 inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--eval-metric', default='map', help='Evaluation metric for describe quantitation error')
    args = parser.parse_args()
    return args


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def save_calib_model(model, output_model_path):
    # Save calibrated checkpoint
    print('Saving calibrated model to {}... '.format(output_model_path))
    dirname = os.path.dirname(output_model_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    torch.save({'model': deepcopy(de_parallel(model)).half()}, output_model_path)
    

def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

   
    cfg = Config.fromfile(args.config)
    class_names = cfg.get('class_name')
    if class_names is None:
        class_names = get_classes('coco')

    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline)
    pre_pipeline = preprocess(cfg)

    path.mkdir_or_exist(args.out_dir)

    # 加载torch模型 考虑pre_pipeline 追溯源码后暂时不需要考虑
    torch_config = Config.fromfile(args.config)
    if 'init_cfg' in torch_config.model.backbone:
        torch_config.model.backbone.init_cfg = None
    device = torch.device(args.device)
    torch_model = init_detector(torch_config, args.torch_checkpoint, device=device, cfg_options={})
    

    import torch.nn as nn    

    from projects.easydeploy.model import DeployModel
    postprocess_cfg = ConfigDict(
            pre_top_k=30000,
            keep_top_k=300,
            iou_threshold=0.65,
            score_threshold=0.001,
            backend=2)
    
    # merge bn and switch_to_deploy
    # call switch_to_deploy
    DeployModel(baseModel=torch_model, postprocess_cfg=postprocess_cfg)
    from mmcv.cnn import ConvModule
    from torch.nn import BatchNorm2d
    # merge bn
    def merge_bn(layer:ConvModule):
        if isinstance(layer.norm, BatchNorm2d):
            bn = layer.norm
            conv = layer.conv
            w_conv = conv.weight.data.clone().view(conv.out_channels, -1)
            w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
            # 融合层的权重初始化(W_bn*w_conv(卷积的权重))
            conv.weight.data.copy_(torch.mm(w_bn, w_conv).view(conv.weight.size()))

            if conv.bias is not None:
                b_conv = conv.bias.data
            else:
                b_conv = torch.zeros(conv.weight.size(0)).to(w_conv.device)
                conv.bias = nn.Parameter(b_conv)
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            # 融合层偏差的设置
            conv.bias.data.copy_(torch.matmul(w_bn, b_conv) + b_bn)
            setattr(layer, layer.norm_name, nn.Identity())
        return layer

    for layer in torch_model.modules():
        if isinstance(layer, ConvModule):
            merge_bn(layer)
    
    output_dir = args.out_dir
    model_name = os.path.splitext(os.path.basename(args.config))[0]    
    
    # deploy_model.eval()
    # 转换算子
    quant_config = QuantConfig(num_bits=8, calib_method='histogram', histogram_amax_method='entropy', histogram_amax_percentile=99.99)
    quant_model = quant_model_init(torch_model, quant_config)
    quant_model.to(device)
    print('quant_model:')
    print(quant_model)

    class QuantDataset(Dataset):
        def __init__(self, filepath) -> None:
            super().__init__()
            self.calib_files, _ = get_file_list(filepath)
        def __getitem__(self, index):
            file = self.calib_files[index]
            bgr = mmcv.imread(file, channel_order='bgr', backend='cv2')
            data, samples = test_pipeline(dict(img=bgr, img_id=index)).values()
            data = pre_pipeline(data)[0]
            return data
        def __len__(self):
            return len(self.calib_files)

    quant_dataset = QuantDataset(args.calib_imgs)
    quant_dataloader = DataLoader(quant_dataset, batch_size=32, num_workers=os.cpu_count() // 8 * 8, shuffle=False)

    from self_study_scripts.quant_utils import collect_stats, compute_amax
    with torch.no_grad():
        collect_stats(quant_model, quant_dataloader, num_batches=4)
        compute_amax(quant_model, method=quant_config.histogram_amax_method, percentile=quant_config.histogram_amax_percentile)

    pth_filepath = os.path.join(output_dir, model_name + "_deploy_model.pth")
    save_calib_model(quant_model, pth_filepath)
    zero_scale_fix_(quant_model)


    test_filepaths = get_file_list(args.img)[0]
    EvalMethodDict = {
        "map":partial(quant_sensitivity_analyse_by_map, eval_function=partial(eval_function_by_map, filepaths=test_filepaths)),
        "cosine":partial(eval_function_by_metric_function, dataloader=quant_dataloader, metric_function=partial(cosine, reduction="mean")),
        "mse":partial(eval_function_by_metric_function, dataloader=quant_dataloader, metric_function=lambda x,y:-torch.mean((x - y) ** 2).detach().item()),
        "sqnr":partial(eval_function_by_metric_function, dataloader=quant_dataloader, metric_function=lambda x,y:-sqnr(x, y))
    }
    eval_method = args.eval_metric
    if eval_method in EvalMethodDict:
        eval_function = EvalMethodDict[eval_method]
    else:
        raise RuntimeError("{} is not supported".format(eval_method))

    result = eval_function(quant_model)
    print('quant_sensitivity_analyse')
    print(result)
    os.makedirs(output_dir, exist_ok=True)
    pkl_filepath = os.path.join(output_dir, 'quant_sensitivity_analyse_result.pkl')
    import pickle
    with open(pkl_filepath, 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()
