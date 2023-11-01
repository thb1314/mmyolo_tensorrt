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
import pickle
from copy import deepcopy
from self_study_scripts.utils import de_parallel
from mmdet.apis import inference_detector, init_detector
from torch.utils.data import DataLoader, Dataset


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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('torch_checkpoint', help='Pytorch Checkpoint file')
    parser.add_argument('pickle_dir', help='Pickle Directory')
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

    pkl_dir = args.pickle_dir
    pkl_filepath = os.path.join(pkl_dir, 'quant_sensitivity_analyse_result.pkl')
    with open(pkl_filepath, 'rb') as f:
        quant_sensitivity_analyse_result = pickle.load(f)
    
    # sensitivity_analyse_result = quant_sensitivity_analyse_result[0]
    # quant_sensitivity_analyse_result = [(item[0], float(item[1][0] - sensitivity_analyse_result[1][0])) for item in quant_sensitivity_analyse_result[1:]]

    quant_sensitivity_analyse_result = sorted(quant_sensitivity_analyse_result, key=lambda x: x[1])
    sensitive_layers_list = [item[0] for item in quant_sensitivity_analyse_result[:6]]
    print('sensitive_layers_list')
    print(sensitive_layers_list)
    
    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
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
    deploy_model = DeployModel(baseModel=torch_model, postprocess_cfg=postprocess_cfg)
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
    quant_model = quant_model_init(torch_model, quant_config, sensitive_layers_skip=True, sensitive_layers_list=sensitive_layers_list)
    quant_model.to(device)
    print('quant_model:')
    print(quant_model)

    class QuantDataset(Dataset):
        def __init__(self) -> None:
            super().__init__()
            self.calib_files, _ = get_file_list(args.calib_imgs)
        def __getitem__(self, index):
            file = self.calib_files[index]
            bgr = mmcv.imread(file, channel_order='bgr', backend='cv2')
            data, samples = test_pipeline(dict(img=bgr, img_id=index)).values()
            data = pre_pipeline(data)[0]
            return data
        def __len__(self):
            return len(self.calib_files)
    
    quant_dataset = QuantDataset()
    quant_dataloader = DataLoader(quant_dataset, batch_size=32, num_workers=os.cpu_count() // 8 * 8, shuffle=False)
    # It is a bit slow since we collect histograms on CPU
    from self_study_scripts.quant_utils import collect_stats, compute_amax
    with torch.no_grad():
        collect_stats(quant_model, quant_dataloader)
        compute_amax(quant_model, method=quant_config.histogram_amax_method, percentile=quant_config.histogram_amax_percentile)

    output_dir = args.out_dir
    model_name = os.path.splitext(os.path.basename(args.config))[0]
    pth_filepath = os.path.join(output_dir, model_name + "_deploy_model.pth")
    save_calib_model(quant_model, pth_filepath)

    zero_scale_fix_(quant_model)
    # 导出onnx模型 仅仅是为了得到量化参数
    input_shape_dict={'input': [1, 3, 640, 640]}
    onnx_filepath = os.path.join(output_dir, model_name + "_deploy_model.onnx")
    
    deploy_model = DeployModel(baseModel=quant_model, postprocess_cfg=postprocess_cfg)
    device = next(deploy_model.parameters()).device
    dynamic_axes={
            'images': {
                0: 'batch_size',
            },
            'num_dets': {
                0: 'batch_size',
            },
            'boxes': {
                0: 'batch_size'
            },
            'scores': {
                0: 'batch_size'
            },
            'labels': {
                0: 'batch_size'
            }
    }
    
    from pytorch_quantization import nn as quant_nn
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    output_names = ['num_dets', 'boxes', 'scores', 'labels']
    torch.onnx.export(deploy_model,
                      torch.rand(*input_shape_dict['input']).to(device),
                      onnx_filepath,
                      verbose=False,
                      opset_version=13,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=output_names,
                      dynamic_axes=dynamic_axes
    )

    import onnxsim
    import onnx
    onnx_model = onnx.load(onnx_filepath)
    onnx_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(onnx_model, onnx_filepath)
    else:
        print('onnxsim failed')

    import onnx_graphsurgeon as gs
    # 对所有的输入节点的graph

    
    trt_runner = TRTRunner(onnx_filepath, max_batch_size=max_batch_size, enable_fp16=True, int8_calibrator=None,
                           dynamic_range_file=None, enable_int8=True)

    # restore torch model
    torch_model = init_detector(torch_config, args.torch_checkpoint, device=device, cfg_options={})
    
    # get file list
    files, _ = get_file_list(args.img)
    # start detector inference
    len_files = len(files)
    image_det_result = dict()
    image_torch_det_result = dict()
    progress_bar = ProgressBar((len_files + max_batch_size - 1) // max_batch_size)

    for start_index in range(0, len_files, max_batch_size):
        end_index = min(len_files, start_index + max_batch_size)
        data_list = []
        shape_list = []
        pad_param_list = []
        scale_factor_list = []
        file_list = []

        for i, file in enumerate(files[start_index:end_index]):
            torch_result = inference_detector(quant_model, file)
            torch_labels = torch_result.pred_instances.labels
            torch_scores = torch_result.pred_instances.scores
            torch_bboxes = torch_result.pred_instances.bboxes
            image_torch_det_result[file] = {
                'scores':torch_scores.cpu().numpy().tolist(),
                'bboxes':torch_bboxes.cpu().numpy().tolist(),
                'labels':torch_labels.cpu().numpy().tolist()
            }

            bgr = mmcv.imread(file, channel_order='bgr', backend='cv2')
            data, samples = test_pipeline(dict(img=bgr, img_id=(start_index + i))).values()
            pad_param = samples.get('pad_param',
                                    np.array([0, 0, 0, 0], dtype=np.float32))
            h, w = samples.get('ori_shape', bgr.shape[:2])
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
    
    with open(os.path.join(args.out_dir, 'torch_result.json'), 'w', encoding='utf-8') as fp:
        json.dump(image_torch_det_result, fp, ensure_ascii=False)


if __name__ == '__main__':
    main()
