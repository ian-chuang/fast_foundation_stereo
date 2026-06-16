import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Union, Any

import cv2
import numpy as np
import torch
import tyro
import yaml
import viser

# PyOrbbecSDK
from pyorbbecsdk import (
    Config,
    Context,
    OBFormat,
    OBSensorType,
    OBFrameType,
    Pipeline,
    Device,
    VideoFrame,
    OBPropertyID,
)

# ---------------------------------------------------------
# Standalone Inference/Viz Helpers
# ---------------------------------------------------------

code_dir = os.path.dirname(os.path.realpath(__file__))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_imagenet(img_uint8: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization: (img/255 - mean) / std."""
    return ((img_uint8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD

def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output=None):
    """
    @disp: np array (H,W)
    @invalid_thres: > thres is invalid
    """
    if other_output is None:
        other_output = {}
    disp = disp.copy()
    H,W = disp.shape[:2]
    invalid_mask = disp>=invalid_thres
    if (invalid_mask==0).sum()==0:
        other_output['min_val'] = None
        other_output['max_val'] = None
        return np.zeros((H,W,3))
    if min_val is None:
        min_val = disp[invalid_mask==0].min()
    if max_val is None:
        max_val = disp[invalid_mask==0].max()
    other_output['min_val'] = min_val
    other_output['max_val'] = max_val
    vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
    if cmap is None:
        vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
    else:
        vis = cmap(vis.astype(np.uint8))[...,:3]*255
    if invalid_mask.any():
        vis[invalid_mask] = 0
    return vis.astype(np.uint8)

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

# ---------------------------------------------------------
# Orbbec Camera Helpers
# ---------------------------------------------------------

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height : height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4 :].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)

def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)

def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

def frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        return i420_to_bgr(data, width, height)
    elif color_format == OBFormat.NV12:
        return nv12_to_bgr(data, width, height)
    elif color_format == OBFormat.NV21:
        return nv21_to_bgr(data, width, height)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    
    print(f"Unsupported color format: {color_format}")
    return None

def depth2xyzmap(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Calculates 3D map from depth map and camera intrinsics."""
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class Args:
    model_path: str = f"{code_dir}/../output/fast_foundationstereo.engine"
    """Path to TensorRT engine or ONNX model."""
    
    cfg_path: str = f"{code_dir}/../output/fast_foundationstereo.yaml"
    """Path to the yaml configuration file for model."""
    
    host: str = "0.0.0.0"
    """Host IP for viser server."""
    
    port: int = 8080
    """Port for viser server."""
    
    cam_width: int = 640
    """Orbbec camera resolution width."""
    
    cam_height: int = 480
    """Orbbec camera resolution height."""
    
    zfar: float = 100.0
    """Max depth to include in point cloud."""

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args: Args):
    torch.autograd.set_grad_enabled(False)
    
    # 1. Start Viser Server
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Running viser server on http://{args.host}:{args.port}")
    
    # GUI for FPS info and Controls
    with server.gui.add_folder("Info & Controls"):
        gui_fps = server.gui.add_number("FPS", initial_value=0.0, disabled=True)
        gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.01)
        gui_zfar = server.gui.add_slider("Z-Far (m)", min=0.01, max=100.0, step=0.1, initial_value=args.zfar)
        gui_filter_invisible = server.gui.add_checkbox("Filter Invisible BG", initial_value=True)

    with server.gui.add_folder("2D Views"):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        gui_left_img = server.gui.add_image(dummy_img, format="jpeg")
        gui_right_img = server.gui.add_image(dummy_img, format="jpeg")
        gui_depth_img = server.gui.add_image(dummy_img, format="jpeg")
    
    # 2. Setup FFS Model
    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg['image_size']
    print(f'Model target resolution: {target_h} x {target_w}')
    
    print(f'Loading model: {args.model_path}')
    if args.model_path.endswith('.onnx'):
        runner = OnnxRuntimeRunner(args.model_path)
    else:
        runner = SingleEngineTrtRunner(args.model_path)
        
    # 3. Setup Orbbec Camera
    print("Setting up Orbbec Pipeline...")
    ctx = Context()
    pipeline = Pipeline()
    device = pipeline.get_device()

    # Load high rate preset configurations
    preset_list = device.get_available_preset_list()
    if preset_list.get_count() > 5:
        device.load_preset(preset_list[5]) # Arbitrary common preset in test_orbbec_ffs.py

    device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 30)

    config = Config()
    pipeline.enable_frame_sync()
    
    # Enable streams
    left_profiles = pipeline.get_stream_profile_list(OBSensorType.LEFT_COLOR_SENSOR)
    left_color_profile = left_profiles.get_video_stream_profile(args.cam_width, args.cam_height, OBFormat.RGB, 60)
    config.enable_stream(left_color_profile)
    
    right_profiles = pipeline.get_stream_profile_list(OBSensorType.RIGHT_COLOR_SENSOR)
    right_color_profile = right_profiles.get_video_stream_profile(args.cam_width, args.cam_height, OBFormat.RGB, 60)
    config.enable_stream(right_color_profile)
    
    baseline = device.get_baseline().baseline / 1000.0  # meters
    print(f"Baseline: {baseline:.4f} m")

    # Get intrinsics and compute rectification maps
    left_intrinsics = left_color_profile.get_intrinsic()
    left_distortion = left_color_profile.get_distortion()
    
    # Construct maps
    left_camera_matrix = np.array([
        [left_intrinsics.fx, 0, left_intrinsics.cx],
        [0, left_intrinsics.fy, left_intrinsics.cy],
        [0, 0, 1]
    ])
    left_dist_coeffs = np.array([
        left_distortion.k1, left_distortion.k2, left_distortion.p1, 
        left_distortion.p2, left_distortion.k3, left_distortion.k4,
        left_distortion.k5, left_distortion.k6
    ])
    
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_dist_coeffs, None, left_camera_matrix,
        (args.cam_width, args.cam_height), cv2.CV_16SC2
    )

    right_intrinsics = right_color_profile.get_intrinsic()
    right_distortion = right_color_profile.get_distortion()
    
    right_camera_matrix = np.array([
        [right_intrinsics.fx, 0, right_intrinsics.cx],
        [0, right_intrinsics.fy, right_intrinsics.cy],
        [0, 0, 1]
    ])
    right_dist_coeffs = np.array([
        right_distortion.k1, right_distortion.k2, right_distortion.p1, 
        right_distortion.p2, right_distortion.k3, right_distortion.k4,
        right_distortion.k5, right_distortion.k6
    ])
    
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_dist_coeffs, None, right_camera_matrix,
        (args.cam_width, args.cam_height), cv2.CV_16SC2
    )

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Pipeline start error: {e}")
        return

    # Loop variables
    frame_count = 0
    last_fps_print = time.perf_counter()

    print("Pipeline started. Streaming data to Viser...")

    try:
        while True:
            t_start = time.perf_counter()
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            
            left_frame = frames.get_frame(OBFrameType.LEFT_COLOR_FRAME)
            right_frame = frames.get_frame(OBFrameType.RIGHT_COLOR_FRAME)
            
            if not left_frame or not right_frame:
                continue

            left_frame = left_frame.as_color_frame()
            right_frame = right_frame.as_color_frame()

            if not left_frame or not right_frame:
                continue

            # Decode
            left_img_bgr = frame_to_bgr_image(left_frame)
            right_img_bgr = frame_to_bgr_image(right_frame)
            
            if left_img_bgr is None or right_img_bgr is None:
                continue

            # Rectify
            left_img = cv2.remap(left_img_bgr, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
            right_img = cv2.remap(right_img_bgr, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)
            
            # RGB conversion for Viser and scaling for inference
            left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            orig_h, orig_w = left_img_rgb.shape[:2]
            fx = target_w / orig_w
            fy = target_h / orig_h

            if fx != 1 or fy != 1:
                img0_resized = cv2.resize(left_img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                img1_resized = cv2.resize(right_img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                img0_resized = left_img_rgb
                img1_resized = right_img_rgb
                
            H, W = img0_resized.shape[:2]

            # Inference
            img0_norm = normalize_imagenet(img0_resized)
            img1_norm = normalize_imagenet(img1_resized)

            t_left  = torch.as_tensor(img0_norm).cuda().float()[None].permute(0, 3, 1, 2)
            t_right = torch.as_tensor(img1_norm).cuda().float()[None].permute(0, 3, 1, 2)

            outputs = runner({'left_image': t_left, 'right_image': t_right})
            disp = outputs['disparity']

            # Scale disparity back to original resolution scale
            disp = disp.float().cpu().numpy().reshape(H, W).clip(0, None) * (1.0 / fx)

            # 2d Disparity Visualization
            vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO) # shape: (target_h, target_w, 3) BGR
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            
            # 3d Point Cloud Generation
            K = left_camera_matrix.copy()
            K[:2] *= np.array([fx, fy], dtype=np.float32)[:, np.newaxis]
            
            # Avoid division by zero
            safe_disp = disp.copy()
            safe_disp[safe_disp < 1e-3] = 1e-3
            depth = K[0, 0] * baseline / safe_disp
            
            # Filter invisible background
            if gui_filter_invisible.value:
                _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                invalid = (xx - disp) < 0
                depth[invalid] = np.inf

            xyz_map = depth2xyzmap(depth, K) # (H, W, 3)
            
            pts = xyz_map.reshape(-1, 3)
            colors = left_img_rgb.reshape(-1, 3)
            
            # Mask valid points
            current_zfar = gui_zfar.value
            valid_mask = (pts[:, 2] > 0) & (pts[:, 2] <= current_zfar) & np.isfinite(pts[:, 2])
            pts = pts[valid_mask]
            colors = colors[valid_mask]

            # In Viser, z is typically up and -y is forward (depending on camera setup)
            # Standard computer vision camera: z is forward, x is right, y is down.
            # To correct for viser default view: Transform camera coordinates to canonical coordinates.
            viser_pts = pts.copy()
            viser_pts[:, 1] = -pts[:, 1]  # Flip Y
            viser_pts[:, 2] = -pts[:, 2]  # Flip Z to match Viser standard view setup (optional)

            # Send visualizations to Viser GUI and Scene
            gui_left_img.image = left_img_rgb
            gui_right_img.image = right_img_rgb
            gui_depth_img.image = vis_rgb
            
            server.scene.add_point_cloud("PointCloud", points=viser_pts, colors=colors, point_size=gui_point_size.value)

            # Calculate FPS
            t_end = time.perf_counter()
            elapsed = t_end - last_fps_print
            frame_count += 1
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                gui_fps.value = float(fps)
                frame_count = 0
                last_fps_print = t_end
                
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
