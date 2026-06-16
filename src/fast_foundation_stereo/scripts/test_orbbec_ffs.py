# ******************************************************************************
#  Orbbec Stereo Camera - Synchronized Left/Right Color Frames Viewer
#
#  What you will learn:
#    1. How to enable left and right color streams simultaneously
#    2. How to synchronize frames between stereo cameras
#    3. How to display stereo color frames side by side
#
#  Keyboard:
#    Q / ESC — Quit
#
#  Dependencies: numpy, opencv-python, pyorbbecsdk
#
#  Run:
#    python test_orbbec.py
# ******************************************************************************

import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    Context,
    OBFormat,
    OBSensorType,
    OBFrameType,
    Pipeline,
    OBPropertyID,
)

from typing import Any, Optional, Union

import cv2
import numpy as np

from pyorbbecsdk import (
    Device,
    FormatConvertFilter,
    OBConvertFormat,
    OBFormat,
    OBSensorType,
    VideoFrame,
)


def is_astra_mini_device(vid: int, pid: int) -> bool:
    if (vid == 0x2BC5) and (pid == 0x069D or pid == 0x065B or pid == 0x065E):
        return True
    return False


def is_gemini305_device(vid: int, pid: int) -> bool:
    if (vid == 0x2BC5) and (pid in (0x0840, 0x0841, 0x0842, 0x0843, 0x0845)):
        return True
    return False


def is_gemini305g_device(vid: int, pid: int, connection_type: str) -> bool:
    return is_gemini305_device(vid, pid) and connection_type == "GMSL2"


def is_lidar_device(device: Device) -> bool:
    sensor_list = device.get_sensor_list()
    count = sensor_list.get_count()
    for index in range(count):
        sensor_type = sensor_list.get_sensor_by_index(index).get_type()
        if sensor_type == OBSensorType.LIDAR_SENSOR:
            return True
    return False


def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    yuyv = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    return bgr_image


def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    uyvy = frame.reshape((height, width, 2))
    bgr_image = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    return bgr_image


def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height : height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4 :].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image


def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image


def determine_convert_format(frame: VideoFrame):
    if frame.get_format() == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    elif frame.get_format() == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    elif frame.get_format() == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    elif frame.get_format() == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    elif frame.get_format() == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    elif frame.get_format() == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    else:
        return None


def frame_to_rgb_frame(frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    print("covert format: {}".format(convert_format))
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print("Convert {} to RGB failed".format(frame.get_format()))
    return rgb_frame


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image


# --- Configuration Constants ---
ESC_KEY = 27

def main():

    ctx = Context()
    # Method 4
    pipeline = Pipeline()
    # Get the device by pipeline.
    device = pipeline.get_device()

    preset_list = device.get_available_preset_list()
    for i in range(preset_list.get_count()):
        print(f"preset {i}: {preset_list.get_name_by_index(i)}")
    preset_name = preset_list[5]
    device.load_preset(preset_name)

    res_cfg_list = device.get_available_preset_resolution_config_list()
    for i in range(res_cfg_list.get_count()):
        cfg = res_cfg_list.get_preset_resolution_ratio_config(i)
        print(f"res cfg {i}: {cfg.width}x{cfg.height}")

    device.set_preset_resolution_config(res_cfg_list.get_preset_resolution_ratio_config(43))

    device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
    device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 30)

    # config = Config()
    # sensor_list = device.get_sensor_list()
    # # 4.Enable all available video streams
    # for sensor in range(len(sensor_list)):
    #     sensor_type = sensor_list[sensor].get_type()

    #     print(f"Enabling sensor type: {sensor_type}")
    #     config.enable_stream(sensor_type)

    width = 640
    height = 480

    config = Config()
    pipeline.enable_frame_sync()
    profiles = pipeline.get_stream_profile_list(OBSensorType.LEFT_COLOR_SENSOR)
    left_color_profile = profiles.get_video_stream_profile(width, height, OBFormat.RGB, 60)
    config.enable_stream(left_color_profile)
    profiles = pipeline.get_stream_profile_list(OBSensorType.RIGHT_COLOR_SENSOR)
    right_color_profile = profiles.get_video_stream_profile(width, height, OBFormat.RGB, 60)
    config.enable_stream(right_color_profile)


    print(f"baseline: {device.get_baseline().baseline} mm")
    baseline = device.get_baseline().baseline / 1000.0  # convert mm to meters

    # Get color internala parameters
    left_color_intrinsics = left_color_profile.get_intrinsic()
    print("left_color_intrinsics  {}".format(left_color_intrinsics))
    # Get color distortion parameter
    left_color_distortion = left_color_profile.get_distortion()
    print("left_color_distortion  {}".format(left_color_distortion))

    sx = width / left_color_intrinsics.width
    sy = height / left_color_intrinsics.height
    cx = left_color_intrinsics.cx * sx
    cy = left_color_intrinsics.cy * sy
    fx = left_color_intrinsics.fx * sx
    fy = left_color_intrinsics.fy * sy
    left_camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    left_camera_distortion = np.array([
        left_color_distortion.k1, 
        left_color_distortion.k2, 
        left_color_distortion.p1, 
        left_color_distortion.p2, 
        left_color_distortion.k3,
        left_color_distortion.k4,
        left_color_distortion.k5,
        left_color_distortion.k6,    
    ])

    right_color_intrinsics = right_color_profile.get_intrinsic()
    print("right_color_intrinsics  {}".format(right_color_intrinsics))
    # Get color distortion parameter
    right_color_distortion = right_color_profile.get_distortion()
    print("right_color_distortion  {}".format(right_color_distortion))

    sx = width / right_color_intrinsics.width
    sy = height / right_color_intrinsics.height
    cx = right_color_intrinsics.cx * sx
    cy = right_color_intrinsics.cy * sy
    fx = right_color_intrinsics.fx * sx
    fy = right_color_intrinsics.fy * sy
    right_camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    right_camera_distortion = np.array([
        right_color_distortion.k1, 
        right_color_distortion.k2, 
        right_color_distortion.p1, 
        right_color_distortion.p2, 
        right_color_distortion.k3,
        right_color_distortion.k4,
        right_color_distortion.k5,
        right_color_distortion.k6,    
    ])

    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix,
        left_camera_distortion,
        None,
        left_camera_matrix,  # output camera matrix
        (width, height),
        cv2.CV_16SC2
    )

    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix,
        right_camera_distortion,
        None,
        right_camera_matrix,
        (width, height),
        cv2.CV_16SC2
    )

                                   

    
    # Start pipeline
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Pipeline start error: {e}")
        return
    






    # fast foundation stereo

    # parser = argparse.ArgumentParser(
    #     description='Run Fast FoundationStereo with ONNX Runtime or TensorRT')
    # parser.add_argument('--model_dir', type=str,
    #                     default=f'{code_dir}/output_single_onnx',
    #                     help='Directory containing .onnx/.engine + config.yaml')
    # parser.add_argument('--model_file', type=str, default='',
    #                     help='Explicit path to .onnx or .engine file (overrides auto-search)')
    # parser.add_argument('--left_file', type=str,
    #                     default=f'{code_dir}/../demo_data/left.png')
    # parser.add_argument('--right_file', type=str,
    #                     default=f'{code_dir}/../demo_data/right.png')
    # parser.add_argument('--intrinsic_file', type=str,
    #                     default=f'{code_dir}/../demo_data/K.txt',
    #                     help='Camera intrinsic matrix and baseline file')
    # parser.add_argument('--out_dir', type=str,
    #                     default=f'{code_dir}/../output_demo')
    # parser.add_argument('--remove_invisible', type=int, default=1)
    # parser.add_argument('--denoise_cloud', type=int, default=1)
    # parser.add_argument('--denoise_nb_points', type=int, default=30)
    # parser.add_argument('--denoise_radius', type=float, default=0.03)
    # parser.add_argument('--get_pc', type=int, default=1,
    #                     help='Generate and save point cloud')
    # parser.add_argument('--zfar', type=float, default=100,
    #                     help='Max depth (m) to include in point cloud')
    # args = parser.parse_args()
    # python scripts/run_demo_single_trt.py \
    # --model_dir output/ \
    # --left_file demo_data/left.png \
    # --right_file demo_data/right.png \
    # --intrinsic_file demo_data/K.txt \
    # --out_dir output_demo/ \
    # --get_pc 1 \
    # --remove_invisible 0 \
    # --denoise_cloud 1 \
    # --zfar 100

    model_path = "/home/ianchuang/Projects/Fast-FoundationStereo/output/fast_foundationstereo.engine"
    cfg_path = "/home/ianchuang/Projects/Fast-FoundationStereo/output/fast_foundationstereo.yaml"
    import torch
    from fast_foundation_stereo.scripts.run_demo_single_trt import (
        OnnxRuntimeRunner, SingleEngineTrtRunner, normalize_imagenet,
        set_logging_format, set_seed, vis_disparity,
        depth2xyzmap, toOpen3dCloud, o3d,
    )
    import os

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    import yaml
    import logging
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg['image_size']
    logging.info(f'Model target resolution: {target_h} x {target_w}')

    

    # ── Load model (ONNX Runtime or TensorRT) ────────────────────────────
    logging.info(f'Loading model: {model_path}')
    if model_path.endswith('.onnx'):
        runner = OnnxRuntimeRunner(model_path)
    else:
        runner = SingleEngineTrtRunner(model_path)







    print("\nStereo Color Frames Viewer")
    print("Q / ESC — Quit\n")
    window_name = "Left/Right Stereo Color Frames  |  Q/ESC = quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)
    depth_window_name = "Depth (Disparity)  |  Q/ESC = quit"
    cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(depth_window_name, 640, 480)
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window(window_name="PointCloud", width=960, height=720)

    pcd = o3d.geometry.PointCloud()
    o3d_vis.add_geometry(pcd)

    opt = o3d_vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    import time

    frame_count = 0
    last_fps_print = time.perf_counter()

    # Frame loop
    while True:
        try:
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

            # Convert frames to BGR images
            left_image = frame_to_bgr_image(left_frame)
            right_image = frame_to_bgr_image(right_frame)
            
            if left_image is None or right_image is None:
                continue

            left_image = cv2.remap(
                left_image,
                left_map1,
                left_map2,
                interpolation=cv2.INTER_LINEAR
            )

            right_image = cv2.remap(
                right_image,
                right_map1,
                right_map2,
                interpolation=cv2.INTER_LINEAR
            )

            # Resize right image to match left if needed
            if left_image.shape != right_image.shape:
                raise ValueError(f"Left and right images have different shapes: {left_image.shape} vs {right_image.shape}")
                right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))







            # ── Read images ───────────────────────────────────────────────────────
            img0 = left_image
            img1 = right_image


            # if img0.ndim == 2:
            #     img0 = np.tile(img0[..., None], (1, 1, 3))
            #     img1 = np.tile(img1[..., None], (1, 1, 3))
            # img0 = img0[..., :3]
            # img1 = img1[..., :3]

            # ── Resize to model resolution (direct stretch) ────────────────────────
            orig_h, orig_w = img0.shape[:2]
            fx = target_w / orig_w
            fy = target_h / orig_h

            if fx != 1 or fy != 1:
                # logging.info(
                #     f'Resizing images: {orig_h}x{orig_w} → {target_h}x{target_w} '
                #     f'(fx={fx:.4f}, fy={fy:.4f})')
                img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            H, W = img0.shape[:2]

            img0_ori = img0.copy()
            img1_ori = img1.copy()
            # logging.info(f'Image size after resize: {img0.shape}')

            # ── Preprocess: ImageNet normalize → NCHW float tensor ────────────────
            img0_norm = normalize_imagenet(img0)
            img1_norm = normalize_imagenet(img1)

            t_left  = torch.as_tensor(img0_norm).cuda().float()[None].permute(0, 3, 1, 2)
            t_right = torch.as_tensor(img1_norm).cuda().float()[None].permute(0, 3, 1, 2)

            # ── Inference ─────────────────────────────────────────────────────────
            # logging.info('Running inference (first run may be slow due to TRT warmup)')
            outputs = runner({'left_image': t_left, 'right_image': t_right})
            disp = outputs['disparity']
            # logging.info('Inference done')

            disp = disp.float().cpu().numpy().reshape(H, W).clip(0, None) * (1.0 / fx)

            # ── Visualise disparity ──────────────────────────────────────────────
            vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
            cv2.imshow(depth_window_name, vis[:, :, ::-1])
        

            # ── Remove invisible pixels ──────────────────────────────────────────
            remove_invisible = True
            if remove_invisible:
                _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                invalid = (xx - disp) < 0
                disp[invalid] = np.inf

            # ── Point cloud generation ───────────────────────────────────────────
            get_pc = True
            if get_pc:
                # with open(args.intrinsic_file, 'r') as f:
                #     lines = f.readlines()
                #     K = (np.array(list(map(float, lines[0].rstrip().split())))
                #         .astype(np.float32).reshape(3, 3))
                #     baseline = float(lines[1])

                K = left_camera_matrix.copy()
                baseline = baseline
                K[:2] *= np.array([fx, fy], dtype=np.float32)[:, np.newaxis]
                depth = K[0, 0] * baseline / disp
                # np.save(f'{args.out_dir}/depth_meter.npy', depth)

                xyz_map = depth2xyzmap(depth, K)
                pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
                pts = np.asarray(pcd.points)
                zfar = 100.0
                keep = (pts[:, 2] > 0) & (pts[:, 2] <= zfar)
                pcd = pcd.select_by_index(np.where(keep)[0])
                # o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
                # logging.info(f'Point cloud saved to {args.out_dir}')
                denoise_cloud = True
                if denoise_cloud:
                    denoise_nb_points = 30
                    denoise_radius = 0.03
                    # logging.info('Denoising point cloud...')
                    _, ind = pcd.remove_radius_outlier(
                        nb_points=denoise_nb_points,
                        radius=denoise_radius)
                    pcd = pcd.select_by_index(ind)
                    # o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', pcd)

                # logging.info('Visualizing point cloud. Press ESC to exit.')
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()
                # vis.add_geometry(pcd)
                # vis.get_render_option().point_size = 1.0
                # vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
                # ctr = vis.get_view_control()
                # ctr.set_front([0, 0, -1])
                # closest = np.asarray(pcd.points)[:, 2].argmin()
                # ctr.set_lookat(np.asarray(pcd.points)[closest])
                # ctr.set_up([0, -1, 0])
                # vis.run()
                # vis.destroy_window()
                pcd.points = o3d.utility.Vector3dVector(xyz_map.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(img0_ori.reshape(-1, 3) / 255.0)

                o3d_vis.update_geometry(pcd)
                o3d_vis.poll_events()
                o3d_vis.update_renderer()












            # Display frames side by side
            stereo_image = np.hstack((left_image, right_image))
            cv2.imshow(window_name, stereo_image)

            frame_count += 1

            now = time.perf_counter()
            elapsed = now - last_fps_print

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                last_fps_print = now

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), ESC_KEY):
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()