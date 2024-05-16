#!/bin/bash
set -e

sudo apt-get update -y && sudo apt-get install build-essential git curl wget zip unzip

echo "Updating CUDA dependencies"
sudo apt-get install -y nvidia-cuda
sudo apt-get install -y cuda-toolkit-11-4
sudo apt-get install -y cuda-libraries-dev-11-4
sudo apt-get install -y cuda-profiler-api-11-4
sudo apt --fix-broken install -o Dpkg::Options::="--force-overwrite"

echo "Installing Torch CUDA"
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v502/pytorch/torch-1.13.0a0+410ce96a.nv22.12-cp38-cp38-linux_aarch64.whl
python3 -m pip install --no-cache $TORCH_INSTALL --upgrade

echo "Installing TorchVision CUDA"
git clone -b v0.14.0 https://github.com/pytorch/vision torchvision-0140
pushd torchvision-0140
python3 setup.py install --user
popd
rm -rf torchvision-0140

echo "Installing TensorRT"
python3 -m pip install numpy
sudo apt-get install python3-libnvinfer-dev
sudo apt-get install tensorrt --reinstall

echo "Exporting CUDA_ROOT, PATH & LD_LIBRARY_PATH"
echo "export CUDA_ROOT=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=$CUDA_ROOT/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_ROOT/targets/aarch64-linux/lib" >> ~/.bashrc
echo "export CPATH=$CPATH:$CUDA_ROOT/targets/aarch64-linux/include" >> ~/.bashrc