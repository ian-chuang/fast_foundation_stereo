from typing import Optional, Tuple
import cv2
import numpy as np
import time
from threading import Event, Lock, Thread
import logging

logger = logging.getLogger(__name__)

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

def i420_to_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height : height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4 :].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_I420)

def nv21_to_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_NV21)

def nv12_to_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_NV12)

def frame_to_rgb_image(frame: VideoFrame) -> Optional[np.ndarray]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    if color_format == OBFormat.RGB:
        return np.resize(data, (height, width, 3))
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)
    elif color_format == OBFormat.MJPG:
        bgr_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr_image is not None:
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return None
    elif color_format == OBFormat.I420:
        return i420_to_rgb(data, width, height)
    elif color_format == OBFormat.NV12:
        return nv12_to_rgb(data, width, height)
    elif color_format == OBFormat.NV21:
        return nv21_to_rgb(data, width, height)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
    
    logger.warning(f"Unsupported color format: {color_format}")
    return None

def depth2xyzmap(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Calculates 3D map from depth map and camera intrinsics."""
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)

class OrbbecStereoColorCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 60, preset_idx: int = 5, use_async: bool = False):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_async = use_async
        
        # Async variables
        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None
        self.frame_lock: Lock = Lock()
        self.latest_frames: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        print("Setting up Orbbec Pipeline...")
        self.ctx = Context()
        self.pipeline = Pipeline()
        self.device = self.pipeline.get_device()

        # Load high rate preset configurations
        preset_list = self.device.get_available_preset_list()
        if preset_list.get_count() > preset_idx:
            # this is the dual color stereo preset for gemini 305
            self.device.load_preset(preset_list[preset_idx]) 

        self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 30)

        self.config = Config()
        self.pipeline.enable_frame_sync()
        
        # Enable streams
        left_profiles = self.pipeline.get_stream_profile_list(OBSensorType.LEFT_COLOR_SENSOR)
        left_color_profile = left_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        self.config.enable_stream(left_color_profile)
        
        right_profiles = self.pipeline.get_stream_profile_list(OBSensorType.RIGHT_COLOR_SENSOR)
        right_color_profile = right_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        self.config.enable_stream(right_color_profile)
        
        self.baseline = self.device.get_baseline().baseline / 1000.0  # meters
        print(f"Baseline: {self.baseline:.4f} m")

        # Get intrinsics and compute rectification maps
        left_intrinsics = left_color_profile.get_intrinsic()
        left_distortion = left_color_profile.get_distortion()
        
        self.left_camera_matrix = np.array([
            [left_intrinsics.fx, 0, left_intrinsics.cx],
            [0, left_intrinsics.fy, left_intrinsics.cy],
            [0, 0, 1]
        ])
        left_dist_coeffs = np.array([
            left_distortion.k1, left_distortion.k2, left_distortion.p1, 
            left_distortion.p2, left_distortion.k3, left_distortion.k4,
            left_distortion.k5, left_distortion.k6
        ])
        
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, left_dist_coeffs, None, self.left_camera_matrix,
            (width, height), cv2.CV_16SC2
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
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            right_camera_matrix, right_dist_coeffs, None, right_camera_matrix,
            (width, height), cv2.CV_16SC2
        )

    def start(self):
        try:
            self.pipeline.start(self.config)
            if self.use_async:
                self._start_read_thread()
        except Exception as e:
            print(f"Pipeline start error: {e}")
            raise e
            
    def stop(self):
        if self.use_async:
            self._stop_read_thread()
        self.pipeline.stop()

    def _process_and_rectify(self, frames) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extracts, converts to RGB, and rectifies frames. Used by both sync and async loops."""
        if not frames:
            return None
            
        left_frame = frames.get_frame(OBFrameType.LEFT_COLOR_FRAME)
        right_frame = frames.get_frame(OBFrameType.RIGHT_COLOR_FRAME)
        
        if not left_frame or not right_frame:
            return None

        left_frame = left_frame.as_color_frame()
        right_frame = right_frame.as_color_frame()

        if not left_frame or not right_frame:
            return None

        # Decode to RGB
        left_img_rgb = frame_to_rgb_image(left_frame)
        right_img_rgb = frame_to_rgb_image(right_frame)
        
        if left_img_rgb is None or right_img_rgb is None:
            return None

        # Rectify
        left_rectified = cv2.remap(left_img_rgb, self.left_map1, self.left_map2, interpolation=cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img_rgb, self.right_map1, self.right_map2, interpolation=cv2.INTER_LINEAR)
            
        return left_rectified, right_rectified
        
    def _read_loop(self):
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                # 1. Fetch raw frames from SDK
                raw_frames = self.pipeline.wait_for_frames(500)
                
                # 2. Process and rectify inside the background thread
                processed_result = self._process_and_rectify(raw_frames)
                
                if processed_result is not None:
                    # 3. Store the finalized RGB rectified images
                    with self.frame_lock:
                        self.latest_frames = processed_result
                    failure_count = 0

            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error processing frame in background thread: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read/process failures.") from e

    def _start_read_thread(self):
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name="OrbbecStereoColorCamera_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frames = None

    def async_read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Immediately returns the most recent pair of rectified RGB frames 
        (left, right) that were processed by the background thread.
        """
        if not self.use_async:
            raise RuntimeError("async_read called but use_async is False")

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError("Read thread is not running.")

        with self.frame_lock:
            return self.latest_frames
        
    def wait_for_frames(self, timeout=1000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Blocks until frames are available, decodes to RGB, rectifies, and returns (left_img, right_img).
        Use this if running synchronously (use_async=False).
        """
        if self.use_async:
            logger.warning("Calling wait_for_frames synchronously while async thread is running might cause frame drops.")
            
        raw_frames = self.pipeline.wait_for_frames(timeout)
        return self._process_and_rectify(raw_frames)


# from typing import Optional, Tuple
# import cv2
# import numpy as np
# import time
# from threading import Event, Lock, Thread
# import logging

# logger = logging.getLogger(__name__)

# # PyOrbbecSDK
# from pyorbbecsdk import (
#     Config,
#     Context,
#     OBFormat,
#     OBSensorType,
#     OBFrameType,
#     Pipeline,
#     Device,
#     VideoFrame,
#     OBPropertyID,
# )

# def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
#     y = frame[0:height, :]
#     u = frame[height : height + height // 4].reshape(height // 2, width // 2)
#     v = frame[height + height // 4 :].reshape(height // 2, width // 2)
#     yuv_image = cv2.merge([y, u, v])
#     return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)

# def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
#     y = frame[0:height, :]
#     uv = frame[height : height + height // 2].reshape(height // 2, width)
#     yuv_image = cv2.merge([y, uv])
#     return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)

# def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
#     y = frame[0:height, :]
#     uv = frame[height : height + height // 2].reshape(height // 2, width)
#     yuv_image = cv2.merge([y, uv])
#     return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

# def frame_to_bgr_image(frame: VideoFrame) -> Optional[np.ndarray]:
#     width = frame.get_width()
#     height = frame.get_height()
#     color_format = frame.get_format()
#     data = np.asanyarray(frame.get_data())
    
#     if color_format == OBFormat.RGB:
#         image = np.resize(data, (height, width, 3))
#         return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     elif color_format == OBFormat.BGR:
#         return np.resize(data, (height, width, 3))
#     elif color_format == OBFormat.YUYV:
#         image = np.resize(data, (height, width, 2))
#         return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
#     elif color_format == OBFormat.MJPG:
#         return cv2.imdecode(data, cv2.IMREAD_COLOR)
#     elif color_format == OBFormat.I420:
#         return i420_to_bgr(data, width, height)
#     elif color_format == OBFormat.NV12:
#         return nv12_to_bgr(data, width, height)
#     elif color_format == OBFormat.NV21:
#         return nv21_to_bgr(data, width, height)
#     elif color_format == OBFormat.UYVY:
#         image = np.resize(data, (height, width, 2))
#         return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    
#     print(f"Unsupported color format: {color_format}")
#     return None

# def depth2xyzmap(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
#     """Calculates 3D map from depth map and camera intrinsics."""
#     h, w = depth_map.shape
#     u, v = np.meshgrid(np.arange(w), np.arange(h))
#     z = depth_map
#     x = (u - K[0, 2]) * z / K[0, 0]
#     y = (v - K[1, 2]) * z / K[1, 1]
#     return np.stack([x, y, z], axis=-1)

# class OrbbecStereoColorCamera:
#     def __init__(self, width: int = 640, height: int = 480, fps: int = 60, preset_idx: int = 5, use_async: bool = False):
#         self.width = width
#         self.height = height
#         self.fps = fps
#         self.use_async = use_async
        
#         # Async read variables
#         self.thread: Optional[Thread] = None
#         self.stop_event: Optional[Event] = None
#         self.frame_lock: Lock = Lock()
#         self.latest_frames: Optional[Tuple[np.ndarray, np.ndarray]] = None
#         self.latest_timestamp: Optional[float] = None
#         self.new_frame_event: Event = Event()
        
#         print("Setting up Orbbec Pipeline...")
#         self.ctx = Context()
#         self.pipeline = Pipeline()
#         self.device = self.pipeline.get_device()

#         # Load high rate preset configurations
#         preset_list = self.device.get_available_preset_list()
#         if preset_list.get_count() > preset_idx:
#             # this is the dual color stereo preset for gemini 305
#             self.device.load_preset(preset_list[preset_idx]) 

#         self.device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
#         self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
#         self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 30)

#         self.config = Config()
#         self.pipeline.enable_frame_sync()
        
#         # Enable streams
#         left_profiles = self.pipeline.get_stream_profile_list(OBSensorType.LEFT_COLOR_SENSOR)
#         left_color_profile = left_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
#         self.config.enable_stream(left_color_profile)
        
#         right_profiles = self.pipeline.get_stream_profile_list(OBSensorType.RIGHT_COLOR_SENSOR)
#         right_color_profile = right_profiles.get_video_stream_profile(width, height, OBFormat.RGB, fps)
#         self.config.enable_stream(right_color_profile)
        
#         self.baseline = self.device.get_baseline().baseline / 1000.0  # meters
#         print(f"Baseline: {self.baseline:.4f} m")

#         # Get intrinsics and compute rectification maps
#         left_intrinsics = left_color_profile.get_intrinsic()
#         left_distortion = left_color_profile.get_distortion()
        
#         # Construct maps
#         self.left_camera_matrix = np.array([
#             [left_intrinsics.fx, 0, left_intrinsics.cx],
#             [0, left_intrinsics.fy, left_intrinsics.cy],
#             [0, 0, 1]
#         ])
#         left_dist_coeffs = np.array([
#             left_distortion.k1, left_distortion.k2, left_distortion.p1, 
#             left_distortion.p2, left_distortion.k3, left_distortion.k4,
#             left_distortion.k5, left_distortion.k6
#         ])
        
#         self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
#             self.left_camera_matrix, left_dist_coeffs, None, self.left_camera_matrix,
#             (width, height), cv2.CV_16SC2
#         )

#         right_intrinsics = right_color_profile.get_intrinsic()
#         right_distortion = right_color_profile.get_distortion()
        
#         right_camera_matrix = np.array([
#             [right_intrinsics.fx, 0, right_intrinsics.cx],
#             [0, right_intrinsics.fy, right_intrinsics.cy],
#             [0, 0, 1]
#         ])
#         right_dist_coeffs = np.array([
#             right_distortion.k1, right_distortion.k2, right_distortion.p1, 
#             right_distortion.p2, right_distortion.k3, right_distortion.k4,
#             right_distortion.k5, right_distortion.k6
#         ])
        
#         self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
#             right_camera_matrix, right_dist_coeffs, None, right_camera_matrix,
#             (width, height), cv2.CV_16SC2
#         )

#     def start(self):
#         try:
#             self.pipeline.start(self.config)
#             if self.use_async:
#                 self._start_read_thread()
#         except Exception as e:
#             print(f"Pipeline start error: {e}")
#             raise e
            
#     def stop(self):
#         if self.use_async:
#             self._stop_read_thread()
#         self.pipeline.stop()
        
#     def _read_loop(self):
#         if self.stop_event is None:
#             raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

#         failure_count = 0
#         while not self.stop_event.is_set():
#             try:
#                 frames = self.wait_for_frames(timeout=500)
#                 if frames is None:
#                     continue
                
#                 capture_time = time.perf_counter()

#                 with self.frame_lock:
#                     self.latest_frames = frames
#                     self.latest_timestamp = capture_time
#                 self.new_frame_event.set()
#                 failure_count = 0

#             except Exception as e:
#                 if failure_count <= 10:
#                     failure_count += 1
#                     logger.warning(f"Error reading frame in background thread for {self}: {e}")
#                 else:
#                     raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

#     def _start_read_thread(self):
#         self._stop_read_thread()

#         self.stop_event = Event()
#         self.thread = Thread(target=self._read_loop, args=(), name=f"OrbbecStereoColorCamera_read_loop")
#         self.thread.daemon = True
#         self.thread.start()

#     def _stop_read_thread(self):
#         if self.stop_event is not None:
#             self.stop_event.set()

#         if self.thread is not None and self.thread.is_alive():
#             self.thread.join(timeout=2.0)

#         self.thread = None
#         self.stop_event = None

#         with self.frame_lock:
#             self.latest_frames = None
#             self.latest_timestamp = None
#             self.new_frame_event.clear()

#     def async_read(self, timeout_ms: float = 200) -> Optional[Tuple[np.ndarray, np.ndarray]]:
#         if not self.use_async:
#             raise RuntimeError("async_read called but use_async is False")

#         if self.thread is None or not self.thread.is_alive():
#             raise RuntimeError("Read thread is not running.")

#         if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
#             return None # Timeout

#         with self.frame_lock:
#             frames = self.latest_frames
#             self.new_frame_event.clear()

#         return frames

#     def read_latest(self, max_age_ms: int = 500) -> Optional[Tuple[np.ndarray, np.ndarray]]:
#         if not self.use_async:
#             raise RuntimeError("read_latest called but use_async is False")

#         if self.thread is None or not self.thread.is_alive():
#             raise RuntimeError("Read thread is not running.")

#         with self.frame_lock:
#             frames = self.latest_frames
#             timestamp = self.latest_timestamp

#         if frames is None or timestamp is None:
#             return None

#         age_ms = (time.perf_counter() - timestamp) * 1e3
#         if age_ms > max_age_ms:
#             return None # frames are too old

#         return frames
        
#     def wait_for_frames(self, timeout=1000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
#         frames = self.pipeline.wait_for_frames(timeout)
#         if not frames:
#             return None
        
#         left_frame = frames.get_frame(OBFrameType.LEFT_COLOR_FRAME)
#         right_frame = frames.get_frame(OBFrameType.RIGHT_COLOR_FRAME)
        
#         if not left_frame or not right_frame:
#             return None

#         left_frame = left_frame.as_color_frame()
#         right_frame = right_frame.as_color_frame()

#         if not left_frame or not right_frame:
#             return None

#         # Decode
#         left_img_bgr = frame_to_bgr_image(left_frame)
#         right_img_bgr = frame_to_bgr_image(right_frame)
        
#         if left_img_bgr is None or right_img_bgr is None:
#             return None

#         # Rectify
#         left_img = cv2.remap(left_img_bgr, self.left_map1, self.left_map2, interpolation=cv2.INTER_LINEAR)
#         right_img = cv2.remap(right_img_bgr, self.right_map1, self.right_map2, interpolation=cv2.INTER_LINEAR)
            
#         return left_img, right_img
