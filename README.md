# Fast Foundation Stereo Setup (CUDA + TensorRT + UV)

---

## 1. Install CUDA Toolkit 12.9.1

Download and install from:

https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

---

## 2. Install TensorRT Tools

```bash
sudo apt install -y \
  libnvinfer-bin=11.0.0.114-1+cuda12.9 \
  libnvinfer11=11.0.0.114-1+cuda12.9 \
  libnvinfer-plugin11=11.0.0.114-1+cuda12.9 \
  libnvinfer-lean11=11.0.0.114-1+cuda12.9 \
  libnvinfer-dispatch11=11.0.0.114-1+cuda12.9 \
  libnvonnxparsers11=11.0.0.114-1+cuda12.9 \
  libnvinfer-vc-plugin11=11.0.0.114-1+cuda12.9
```

---

## 3. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 4. Clone Repository

```bash
git clone https://github.com/ian-chuang/fast_foundation_stereo
cd fast_foundation_stereo
uv sync
```

---

## 5. Run Demo Model (PyTorch)

```bash
uv run python -m fast_foundation_stereo.scripts.run_demo \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output/ \
  --remove_invisible 0 \
  --denoise_cloud 1 \
  --scale 1 \
  --get_pc 1 \
  --valid_iters 8 \
  --max_disp 192 \
  --zfar 100
```

---

## 6. Export ONNX + TensorRT Engine

### 6.1 Export ONNX

```bash
uv run python -m fast_foundation_stereo.scripts.make_single_onnx \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output/ \
  --height 480 \
  --width 640 \
  --valid_iters 8 \
  --max_disp 192
```

### 6.2 Convert ONNX (FP16)

```bash
uv run python -m modelopt.onnx.autocast \
  --onnx_path output/fast_foundationstereo.onnx
```

### 6.3 Build TensorRT Engine

```bash
trtexec \
  --onnx=output/fast_foundationstereo.fp16.onnx \
  --saveEngine=output/fast_foundationstereo.engine
```

---

## 7. Run TensorRT Demo

```bash
uv run python -m fast_foundation_stereo.scripts.run_demo_single_trt \
  --model_dir output/ \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output_demo/ \
  --get_pc 1 \
  --remove_invisible 0 \
  --denoise_cloud 1 \
  --zfar 100
```

---

## 8. Orbbec Camera

### Test camera

```bash
python -m fast_foundation_stereo.scripts.test_orbbec
```

### Run live visualization (Viser)

```bash
python -m fast_foundation_stereo.scripts.orbbec_ffs_viser
```




<!-- install cuda toolkit 12.9.1

https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

install tensorrt tools

sudo apt install -y \
  libnvinfer-bin=11.0.0.114-1+cuda12.9 \
  libnvinfer11=11.0.0.114-1+cuda12.9 \
  libnvinfer-plugin11=11.0.0.114-1+cuda12.9 \
  libnvinfer-lean11=11.0.0.114-1+cuda12.9 \
  libnvinfer-dispatch11=11.0.0.114-1+cuda12.9 \
  libnvonnxparsers11=11.0.0.114-1+cuda12.9 \
  libnvinfer-vc-plugin11=11.0.0.114-1+cuda12.9

install uv 

curl -LsSf https://astral.sh/uv/install.sh | sh

download repo

git clone https://github.com/ian-chuang/fast_foundation_stereo
cd fast_foundation_stereo
uv sync

demo model

uv run python -m fast_foundation_stereo.scripts.run_demo --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100

setup tensorrt model

uv run python -m fast_foundation_stereo.scripts.make_single_onnx --model_dir weights/23-36-37/model_best_bp2_serialize.pth --save_path output/ --height 480 --width 640 --valid_iters 8 --max_disp 192

uv run python -m modelopt.onnx.autocast --onnx_path output/fast_foundationstereo.onnx 

trtexec --onnx=output/fast_foundationstereo.fp16.onnx --saveEngine=output/fast_foundationstereo.engine 

run tensorrt model

uv run python -m fast_foundation_stereo.scripts.run_demo_single_trt --model_dir output/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output_demo/ --get_pc 1 --remove_invisible 0 --denoise_cloud 1 --zfar 100

run orbbec camera

python -m fast_foundation_stereo.scripts.test_orbbec

run orbbec camera fast foundation stereo viser visualization

python -m fast_foundation_stereo.scripts.orbbec_ffs_viser -->



<!-- uv add \
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
  xformers -->