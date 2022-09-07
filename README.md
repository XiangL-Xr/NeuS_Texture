# 基于神经渲染的自动化三维重建API

## step 01: colmap config
    参考链接：http://colmap.github.io/
 
 ### a. dependency packages install
  $ sh apt_install.sh

 ### b. ceres-solver-1.14.0 install
  $ 下载地址: https://codeload.github.com/ceres-solver/ceres-solver/tar.gz/refs/tags/1.14.0   
  $ tar -zxvf ceres-solver-1.14.0.tar.gz  
  $ cd ceres-solver-1.14.0    
  $ mkdir build   
  $ cd build   
  $ cmake ..    
  $ make -j8 
  $ make install    

 ### c. colmap-3.8 install
  $ git clone https://github.com/colmap/colmap.git  
  $ cd colmap  
  $ git checkout dev   
  $ mkdir build  
  $ cd build    
  $ cmake ..    
  $ make -j8  
  $ make install   
  $ colmap -h    

## step 02: NeuS_Texture config 
  $ conda create -n neus_texture python=3.8   
  $ conda activate neus_texture    
  $ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html  
  $ pip install -r requirements.txt  
  $ pip install ninja imageio PyOpenGL glfw xatlas gdown  
  $ pip install git+https://github.com/NVlabs/nvdiffrast/  
  $ pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch  
  $ imageio_download_bin freeimage  

## step 03: API start
  $ git clone https://github.com/XiangL-Xr/NeuS_Texture.git  
  $ cd NeuS_Texture  
  $ python api_runner_v0.1.py  