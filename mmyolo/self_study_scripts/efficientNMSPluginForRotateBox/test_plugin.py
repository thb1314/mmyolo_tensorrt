import os
import ctypes
import numpy as np
# cuda: https://nvidia.github.io/cuda-python/
# required for the inference using TRT engine
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from mmcv.ops import nms_rotated
import torch
import time

soFile = "./build/efficient_nms_rotated.so"
epsilon = 1.0e-2
np.random.seed(97)


def filter_top_k(scores, score_thr, topk):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    return scores, labels, keep_idxs


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    
    def free(self):
        self.host = None
        if self.device is not None:
            self.device.free()
            self.device = None
    
    def __del__(self):
        self.free()
    
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(ori_inputs, ori_outputs, engine, context):
    inputs = []
    outputs = []
    bindings = []
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    
    for i, binding in enumerate(engine):
        size = trt.volume(context.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        try:
            if engine.binding_is_input(binding):
                ori_mem = ori_inputs[i]
            else:
                ori_mem = ori_outputs[i - nInput]
        except:
            ori_mem = None
            
        if ori_mem is not None:
            if ori_mem.host.nbytes >= size:
                host_mem = ori_mem.host
                device_mem = ori_mem.device
                # 避免再次释放
                ori_mem.device = None
            else:
                ori_mem.free()
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
        else:
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings


def printArrayInfo(x, description=""):
    print('%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e' % (
        description, str(x.shape), np.mean(x), np.sum(abs(x)), np.var(x), np.max(x), np.min(x), np.sum(np.abs(np.diff(x.reshape(-1))))))
    print("\t", x.reshape(-1)[:10])


def check(a, b, weak=False):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check:", res, "maxAbsDiff:", diff0, "maxRelDiff:", diff1)


def getEfficientNMS_TRT_ForRotateBox_Plugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'EfficientNMS_TRT_ForRotateBox':
            parameterList = []
            parameterList.append(trt.PluginField("background_class", np.int32(-1), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("box_coding", np.int32(0), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("iou_threshold", np.float32(0.1), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("score_threshold", np.float32(0.05), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("max_output_boxes", np.int32(200), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("score_activation", np.int32(0), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("plugin_version", np.array(["1"],dtype=np.string_), trt.PluginFieldType.CHAR))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


use_fp16 = False
trtFile = "./efficient_nms_rotated.plan"

# score threshold 0.01 effciency: torch > tensorrt
# score threshold 0.05 effciency: tensorrt > torch
def run():
    global use_fp16
    stream = cuda.Stream()
    cuda_context = pycuda.autoinit.context
    cuda_context.push()

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 6 << 30

        if builder.platform_has_fast_fp16 and use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        else:
            use_fp16 = False
        if use_fp16:
            input_boxes = network.add_input('input_boxes', trt.DataType.HALF, [-1, 21504, 5])
            input_scores = network.add_input('input_scores', trt.DataType.HALF, [-1, 21504, 15])
        else:
            input_boxes = network.add_input('input_boxes', trt.DataType.FLOAT, [-1, 21504, 5])
            input_scores = network.add_input('input_scores', trt.DataType.FLOAT, [-1, 21504, 15])
        profile.set_shape(input_boxes.name, [1, 21504, 5], [1, 21504, 5], [1, 21504, 5])
        profile.set_shape(input_scores.name, [1, 21504, 15], [1, 21504, 15], [1, 21504, 15])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2([input_boxes, input_scores], getEfficientNMS_TRT_ForRotateBox_Plugin())
        
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
        for i in range(pluginLayer.num_outputs):
            output_name_layer = pluginLayer.get_output(i)
            output_name_layer.name = output_names[i]
            network.mark_output(output_name_layer)
        if use_fp16:
            pluginLayer.precision = trt.float16
            for i in range(pluginLayer.num_outputs):
                pluginLayer.set_output_type(i, trt.float16)
                network.get_output(i).dtype = trt.float16
        
        engineString = builder.build_serialized_network(network, config)
        if engineString is None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    # 动态输入
    context = engine.create_execution_context()
    shape1 = (1, 21504, 5)
    shape2 = (1, 21504, 15)

    context.set_binding_shape(0, shape1)
    context.set_binding_shape(1, shape2)
    context.set_optimization_profile_async(0, stream.handle)
        
    inputs, outputs, bindings = allocate_buffers(None, None, engine, context)
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    
    boxes = torch.load('boxes.pkl', map_location='cpu')
    scores = torch.load('scores.pkl', map_location='cpu')
    
    for count_iter in range(10):
        start_time = time.perf_counter()
        torch_boxes = boxes.cuda()[0]
        torch_scores = scores.cuda()[0]
        torch_scores, torch_labels, torch_keep_idxs = filter_top_k(torch_scores, 0.05, 2000)
        torch_boxes = torch_boxes[torch_keep_idxs]
        torch_boxes, keep_idx = nms_rotated(torch_boxes, torch_scores, iou_threshold=0.1, labels=torch_labels)
        
    total_time_span = 0.0
    for count_iter in range(10):
        start_time = time.perf_counter()
        torch_boxes = boxes.cuda()[0]
        torch_scores = scores.cuda()[0]
        torch_scores, torch_labels, torch_keep_idxs = filter_top_k(torch_scores, 0.05, 2000)
        torch_boxes = torch_boxes[torch_keep_idxs]
        torch_boxes, keep_idx = nms_rotated(torch_boxes, torch_scores, iou_threshold=0.1, labels=torch_labels)
        torch_labels = torch_labels[keep_idx]
        torch_scores = torch_scores[keep_idx]
        end_time = time.perf_counter()
        time_span = end_time - start_time
        total_time_span += time_span
    total_time_span /= 10
    print('torch time span: %.02f ms' % (total_time_span * 1000,))
    # print('torch nms result', (torch_boxes, torch_labels, torch_scores))

    data_list = [boxes.numpy(), scores.numpy()]
    
    # Do inference
    total_time_span = 0
    for count_iter in range(10):
        start_time = time.perf_counter()
        for i in range(nInput):
            np_type = trt.nptype(engine.get_binding_dtype(i))
            inputs[i].host = np.ascontiguousarray(data_list[i].astype(np_type))
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end_time = time.perf_counter()
        time_span = end_time - start_time
        total_time_span += time_span

    total_time_span = 0
    for count_iter in range(10):
        start_time = time.perf_counter()
        for i in range(nInput):
            np_type = trt.nptype(engine.get_binding_dtype(i))
            inputs[i].host = np.ascontiguousarray(data_list[i].astype(np_type))
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end_time = time.perf_counter()
        time_span = end_time - start_time
        total_time_span += time_span
    total_time_span /= 10
    print('tensorrt time span: %.02f ms' % (total_time_span * 1000,))
    
    trt_outputs_dict = dict()
    
    for i in range(nOutput):
        shape = context.get_binding_shape(nInput + i)
        name = engine.get_binding_name(nInput + i)
        need_size = 1
        for item in shape:
            need_size *= item
        trt_outputs_dict[name] = trt_outputs[i][:need_size].reshape(shape)
    
    cuda_context.pop()
    # print('tensorrt output dict', trt_outputs_dict)


if __name__ == '__main__':
    if os.path.exists(trtFile):
        os.remove(trtFile)
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run()
    print("test finish!")