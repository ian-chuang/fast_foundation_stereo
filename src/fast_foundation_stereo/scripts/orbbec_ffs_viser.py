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

from fast_foundation_stereo.engine_trt_runner import create_runner, normalize_imagenet
from fast_foundation_stereo.orbbec_stereo_color import OrbbecStereoColorCamera, depth2xyzmap

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

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class Args:
    model_path: str = f"output/fast_foundationstereo.engine"
    """Path to TensorRT engine or ONNX model."""
    
    cfg_path: str = f"output/fast_foundationstereo.yaml"
    """Path to the yaml configuration file for model."""
    
    host: str = "0.0.0.0"
    """Host IP for viser server."""
    
    port: int = 8080
    """Port for viser server."""
    
    cam_width: int = 640
    """Orbbec camera resolution width."""
    
    cam_height: int = 480
    """Orbbec camera resolution height."""
    
    zfar: float = 0.5
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
        gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.001)
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
    runner = create_runner(args.model_path)
        
    # 3. Setup Orbbec Camera
    camera = OrbbecStereoColorCamera(
        width=args.cam_width, height=args.cam_height,
        use_async=True,  # Use async capture for better performance    
    )
    camera.start()

    # Loop variables
    frame_count = 0
    last_fps_print = time.perf_counter()

    print("Pipeline started. Streaming data to Viser...")

    try:
        while True:
            t_start = time.perf_counter()
            frames = camera.async_read()
            if not frames:
                continue

            left_img_rgb, right_img_rgb = frames
            
            # RGB conversion for Viser and scaling for inference
            # left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            # right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

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
            K = camera.left_camera_matrix.copy()
            K[:2] *= np.array([fx, fy], dtype=np.float32)[:, np.newaxis]
            
            # Avoid division by zero
            safe_disp = disp.copy()
            safe_disp[safe_disp < 1e-3] = 1e-3
            depth = K[0, 0] * camera.baseline / safe_disp
            
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
        camera.stop()

if __name__ == "__main__":
    tyro.cli(main)
