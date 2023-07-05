import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import torch
import os


logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

def GiB(val):
    return val * 1 << 30

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
def allocate_buffers(ori_inputs, ori_outputs, engine, context, stream):
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


def build_engine(onnx_file_path, enable_fp16 = False, max_batch_size = 1, write_engine=True):

    onnx_path = os.path.realpath(onnx_file_path) 
    engine_file_path = ".".join(onnx_path.split('.')[:-1] + ['engine' if not enable_fp16 else 'fp16.engine'])
    print('engine_file_path', engine_file_path)
    G_LOGGER = trt.Logger(trt.Logger.INFO)
    if os.path.exists(engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine, engine_file_path
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, torch.cuda.get_device_properties('cuda:0').total_memory)
        if enable_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {} ...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    # print(parser.get_error(error))
                    print(" parsing error:", parser.get_error(i).code(), "\n",
                        "function name:", parser.get_error(i).func(), "\n",
                        "node:", parser.get_error(i).node(), "\n",
                        "line num:", parser.get_error(i).line(), "\n",
                        "desc:", parser.get_error(i).desc())
                return None, None
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

        # 从onnx中获取输入形状信息
        import onnx
        onnx_model = onnx.load(onnx_file_path)
        input_names = list(onnx_model.graph.input)
        input_name2shape = dict()
        for value_info in input_names:
            input_name2shape[value_info.name] = [getattr(item, 'dim_value', getattr(item, 'dim_param')) for item in value_info.type.tensor_type.shape.dim]
        
        profile = builder.create_optimization_profile()
        for name, shape in input_name2shape.items():
            one_shape = shape[0:]
            one_shape[0] = 1
            max_shape = shape[0:]
            max_shape[0] = max_batch_size
            one_shape = tuple([int(item) for item in one_shape])
            max_shape = tuple([int(item) for item in max_shape])
            profile.set_shape(name, one_shape, max_shape, max_shape)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            return None, None
        print("Completed creating Engine")
        # 保存engine文件
        if write_engine:
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
        with trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine, engine_file_path


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

class TRTRunner(object):
    def __init__(self, engine_or_onnx_path, max_batch_size=1, enable_fp16=False):
        self.engine_path = engine_or_onnx_path
        self.max_batch_size = max_batch_size
        self.enable_fp16 = enable_fp16
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._get_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = None
        self.outputs = None


    def _get_engine(self):
        # If a serialized engine exists, use it instead of building an engine.
        return build_engine(self.engine_path, enable_fp16=self.enable_fp16, max_batch_size=self.max_batch_size, write_engine=True)[0]

    def detect(self, image_np_array, cuda_ctx = pycuda.autoinit.context):
        if cuda_ctx:
            cuda_ctx.push()

        batch_size = image_np_array.shape[0]
        # 动态输入
        origin_inputshape = self.context.get_binding_shape(0)
        origin_inputshape[0] = batch_size
        self.context.set_binding_shape(0, (origin_inputshape))
        self.context.set_optimization_profile_async(0, self.stream.handle)
        
        self.inputs, self.outputs, bindings = allocate_buffers(self.inputs, self.outputs, self.engine, self.context, self.stream)
        np_type = trt.nptype(self.engine.get_binding_dtype(0))
        # Do inference
        self.inputs[0].host = np.ascontiguousarray(image_np_array.astype(np_type))
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=self.inputs, outputs=self.outputs,
                                          stream=self.stream)
        
        if cuda_ctx:
            cuda_ctx.pop()
        
        nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
        nOutput = self.engine.num_bindings - nInput
        trt_outputs_dict = dict()
        
        for i in range(nOutput):
            shape = self.context.get_binding_shape(nInput + i)
            name = self.engine.get_binding_name(nInput + i)
            need_size = 1
            for item in shape:
                need_size *= item
            trt_outputs_dict[name] = trt_outputs[i][:need_size].reshape(shape)
        return trt_outputs_dict
    
    def __call__(self, x):
        return self.detect(x)
    
    def __del__(self):
        del self.inputs
        del self.outputs
        del self.stream
        del self.engine
        del self.context
