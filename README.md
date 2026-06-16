install cuda toolkit 12.9.1

https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network


sudo apt install -y \
  libnvinfer-bin=11.0.0.114-1+cuda12.9 \
  libnvinfer11=11.0.0.114-1+cuda12.9 \
  libnvinfer-plugin11=11.0.0.114-1+cuda12.9 \
  libnvinfer-lean11=11.0.0.114-1+cuda12.9 \
  libnvinfer-dispatch11=11.0.0.114-1+cuda12.9 \
  libnvonnxparsers11=11.0.0.114-1+cuda12.9 \
  libnvinfer-vc-plugin11=11.0.0.114-1+cuda12.9

uv add \
  einops \
  imageio \
  numpy \
  "nvidia-modelopt[torch,onnx]>=0.44.0" \
  omegaconf \
  onnx \
  onnxruntime-gpu \
  onnxscript \
  open3d \
  opencv-contrib-python \
  pyorbbecsdk2 \
  pyyaml \
  scikit-image \
  scipy \
  "tensorrt-cu12==11.0.0.114" \
  "tensorrt-dispatch-cu12==11.0.0.114" \
  "tensorrt-lean-cu12==11.0.0.114" \
  timm \
  "torch==2.9.0" \
  "torchvision==0.24.1" \
  xformers


python scripts/make_single_onnx.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --save_path output/ --height 480 --width 640 --valid_iters 8 --max_disp 192

python -m modelopt.onnx.autocast --onnx_path output/fast_foundationstereo.onnx 

trtexec --onnx=output/fast_foundationstereo.fp16.onnx --saveEngine=output/fast_foundationstereo.engine 

python scripts/run_demo_single_trt.py --model_dir output/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output_demo/ --get_pc 1 --remove_invisible 0 --denoise_cloud 1 --zfar 100



