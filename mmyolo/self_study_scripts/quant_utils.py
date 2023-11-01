from tqdm import tqdm
import torch
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from .utils import set_module, module_quant_disable, get_module, model_quant_disable, module_quant_enable
import copy
from torch.utils.data import DataLoader
import numpy as np



def collect_stats(model:nn.Module, data_loader:DataLoader, num_batches = None):
    """Feed data to the network and collect statistic"""

    device = next(model.parameters()).device
    num_batches = num_batches or len(data_loader)
    # Enable calibrators
    # Enable weight obersever disable activition obersever
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable()
                if name.endswith('_input_quantizer'):
                    module.disable_quant()
                    module.disable_calib()
                elif name.endswith('_weight_quantizer'):
                    module.disable_quant()
                    module.enable_calib()
                else:
                    raise RuntimeError('name error')
            else:
                module.disable()
    
    for i, elements in enumerate(data_loader):
        if isinstance(elements, (tuple, list)):
            tensor = elements[0]
        elif isinstance(elements, dict):
            tensor = list(elements.values())[0]
        else:
            tensor = elements
        model(tensor.to(device))
        break
    
    for name, module in model.named_modules():    
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable()
                if name.endswith('_input_quantizer'):
                    module.disable_quant()
                    module.enable_calib()
                elif name.endswith('_weight_quantizer'):
                    module.disable_quant()
                    module.disable_calib()
                else:
                    raise RuntimeError('name error')
            else:
                module.disable()

    for i, elements in tqdm(enumerate(data_loader), total=num_batches):
        if isinstance(elements, (tuple, list)):
            tensor = elements[0]
        elif isinstance(elements, dict):
            tensor = list(elements.values())[0]
        else:
            tensor = elements
        model(tensor.to(device))
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model:nn.Module, **kwargs):
    # Load Calib result
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(F"{name:40}: {module}")
            if module._calibrator is not None:
                # MinMaxCalib
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()    
                else:
                    # HistogramCalib
                    module.load_calib_amax(**kwargs)
                module.to(device)



def quantable_op_check(k, quantable_ops):
    if quantable_ops is None:
        return True

    if k in quantable_ops:
        return True
    else:
        return False


def skip_sensitive_layers(model, sensitive_layers):
    print('Skip sensitive layers...')
    for name, module in model.named_modules():
        if name in sensitive_layers:
            print(F"Disable {name}")
            module_quant_disable(model, name)


class QuantConfig:

    def __init__(self, num_bits=8, calib_method="max",
                 sensitive_layers_skip = False, sensitive_layers_list=[],
                 calib_batches = 4, histogram_amax_method="entropy",
                 histogram_amax_percentile=99.99) -> None:
        """
        calib_method: 'max', 'histogram'
        histogram_amax_method: 'entropy', 'percentile', 'mse'
        """
        self.num_bits = num_bits
        self.calib_method = calib_method
        self.sensitive_layers_skip = sensitive_layers_skip
        self.sensitive_layers_list = sensitive_layers_list
        self.calib_batches = calib_batches
        self.histogram_amax_method = histogram_amax_method
        self.histogram_amax_percentile = histogram_amax_percentile
        

def quant_model_init(model: nn.Module, quant_conf: QuantConfig, **kw_args):
    sensitive_layers_skip = kw_args.get('sensitive_layers_skip', None)
    sensitive_layers_list = kw_args.get('sensitive_layers_list', [])
    device = next(model.parameters()).device
    model_ptq = copy.deepcopy(model)
    model_ptq.eval()
    model_ptq.to(device)

    conv2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    conv2d_input_default_desc = QuantDescriptor(num_bits=quant_conf.num_bits, calib_method=quant_conf.calib_method)

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=quant_conf.num_bits, calib_method=quant_conf.calib_method)

    for k, m in model_ptq.named_modules():
        if 'reg_convs' in k or 'cls_convs' in k:
            print('not quantized', k, m)
            continue
            
        # if 'reg_preds' in k or 'cls_preds' in k:
        #     print(k, m)
        #     continue
        if 'proj_conv' in k:
            print("Skip Layer {}, module type {}".format(k, m.__class__.__name__))
            continue
        if sensitive_layers_skip is True:
            if k in sensitive_layers_list:
                print("Skip Layer {}, module type {}".format(k, m.__class__.__name__))
                continue

        if isinstance(m, nn.Conv2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              quant_desc_input = conv2d_input_default_desc,
                                              quant_desc_weight = conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model_ptq, k, quant_conv)
            print("quantized {}, module type {}".format(k, m.__class__.__name__))
        elif isinstance(m, nn.ConvTranspose2d):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       quant_desc_input = convtrans2d_input_default_desc,
                                                       quant_desc_weight = convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model_ptq, k, quant_convtrans)
            print("quantized {}, module type {}".format(k, m.__class__.__name__))
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input = conv2d_input_default_desc)
            set_module(model_ptq, k, quant_maxpool2d)
            print("quantized {}, module type {}".format(k, m.__class__.__name__))
        elif isinstance(m, nn.Upsample):
            new_model = nn.Sequential([
                m,
                quant_nn.TensorQuantizer(conv2d_input_default_desc)
            ])
            set_module(model_ptq, k, new_model)
            print("quantized {}, module type {}".format(k, m.__class__.__name__))
        else:
            # module can not be quantized, continue
            print("not quantized {}, module type {}".format(k, m.__class__.__name__))
            continue

    return model_ptq


def zero_scale_fix_(model:nn.Module):
    device = next(model.parameters()).device
    for k, m in model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
            isinstance(m, quant_nn.QuantConvTranspose2d):
            weight_amax = m._weight_quantizer._amax.detach().cpu().numpy()
            ones = np.ones_like(weight_amax)
            print("zero scale number = {}".format(np.sum(weight_amax == 0.0)))
            weight_amax = np.where(weight_amax == 0.0, ones, weight_amax)
            m._weight_quantizer._amax.copy_(torch.from_numpy(weight_amax).to(device))
        else:
            # module can not be quantized, continue
            continue

