import os
from pytorch_quantization import nn as quant_nn
from copy import deepcopy
import torch.nn as nn


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)


def is_parallel(model):
    '''Return True if model's type is DP or DDP, else False.'''
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    '''De-parallelize a model. Return single-GPU model if model's type is DP or DDP.'''
    return model.module if is_parallel(model) else model


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_module(model, submodule_key):
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def module_quant_disable(model, k):
    cur_module = get_module(model, k)
    if hasattr(cur_module, '_input_quantizer'):
        cur_module._input_quantizer.disable()
    if hasattr(cur_module, '_weight_quantizer'):
        cur_module._weight_quantizer.disable()


def module_quant_enable(model, k):
    cur_module = get_module(model, k)
    if hasattr(cur_module, '_input_quantizer'):
        cur_module._input_quantizer.enable()
    if hasattr(cur_module, '_weight_quantizer'):
        cur_module._weight_quantizer.enable()


def model_quant_disable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable()


def model_quant_enable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable()


def model_calib_disable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_calib()


def model_calib_enable(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_calib()


def concat_quant_amax_fuse(ops_list):
    if len(ops_list) <= 1:
        return

    amax = -1
    for op in ops_list:
        if hasattr(op, '_amax'):
            op_amax = op._amax.detach().item()
        elif hasattr(op, '_input_quantizer'):
            op_amax = op._input_quantizer._amax.detach().item()
        else:
            print("Not quantable op, skip")
            return
        print("op amax = {:7.4f}, amax = {:7.4f}".format(op_amax, amax))
        if amax < op_amax:
            amax = op_amax

    print("amax = {:7.4f}".format(amax))
    for op in ops_list:
        if hasattr(op, '_amax'):
            op._amax.fill_(amax)
        elif hasattr(op, '_input_quantizer'):
            op._input_quantizer._amax.fill_(amax)


