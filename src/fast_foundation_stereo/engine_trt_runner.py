import torch
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_imagenet(img_uint8: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization: (img/255 - mean) / std."""
    return ((img_uint8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD

class SingleEngineTrtRunner:
    """Minimal TensorRT runner for a single engine with named I/O."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f'Failed to deserialize TRT engine from {engine_path}. '
                f'This usually means the engine was built with a different '
                f'TensorRT version (yours: {trt.__version__}). '
                f'Rebuild with:  trtexec --onnx=<your .onnx> '
                f'--saveEngine={engine_path} --fp16')
        self.context = self.engine.create_execution_context()

    def _trt_to_torch_dtype(self, dt):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT:  torch.float32,
            trt.DataType.HALF:   torch.float16,
            trt.DataType.BF16:   torch.bfloat16,
            trt.DataType.INT32:  torch.int32,
            trt.DataType.INT8:   torch.int8,
            trt.DataType.BOOL:   torch.bool,
        }
        if dt not in mapping:
            raise RuntimeError(f'Unsupported TRT dtype: {dt}')
        return mapping[dt]

    def __call__(self, inputs: dict) -> dict:
        """Run inference.

        Args:
            inputs: {binding_name: torch.Tensor} for every input tensor.
        Returns:
            {binding_name: torch.Tensor} for every output tensor.
        """
        trt = self.trt

        for name, tensor in inputs.items():
            expected = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                inputs[name] = tensor.to(expected)
            if not inputs[name].is_contiguous():
                inputs[name] = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))

        out_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
               == trt.TensorIOMode.OUTPUT
        ]

        outputs = {}
        for name in out_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device='cuda', dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        assert self.context.execute_async_v3(stream)

        return outputs


class OnnxRuntimeRunner:
    """Run inference via ONNX Runtime (GPU if available, else CPU)."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        print(f'ONNX Runtime providers: {providers}')
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, inputs: dict) -> dict:
        feed = {}
        for name in self.input_names:
            tensor = inputs[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().float().numpy()
            feed[name] = tensor
        raw_outputs = self.session.run(self.output_names, feed)
        outputs = {}
        for name, arr in zip(self.output_names, raw_outputs):
            outputs[name] = torch.as_tensor(arr).cuda()
        return outputs

def create_runner(model_path: str):
    if model_path.endswith('.onnx'):
        return OnnxRuntimeRunner(model_path)
    else:
        return SingleEngineTrtRunner(model_path)
