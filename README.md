# Image-Processing
Digital 

https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems (from ML-Basic)

https://www.hackster.io/shahizat005/getting-started-with-ai-on-jetson-nano-using-arducam-camera-aa34a2?fbclid=IwAR1EBa41sr8g3Ya486A52Dn4L82QND2_GrNnoVVg9QQe-xiyh7DReuel4tY


# PCL install

sudo apt install libpcl-dev

# OpenCV 4.2.0 with CUDA 10.2 and Qt support

## Install dependencies

sudo apt update

sudo apt upgrade

sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall

sudo apt install libjpeg-dev libpng-dev libtiff-dev

sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev

sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev

sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev

sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev

sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils

sudo apt-get install libgtk-3-dev

sudo apt-get install qt5-default

sudo apt-get install python3-dev python3-pip

sudo -H pip3 install -U pip numpy

sudo apt install python3-testresources

sudo apt-get install libtbb-dev

sudo apt-get install libatlas-base-dev gfortran

sudo apt-get install libprotobuf-dev protobuf-compiler

sudo apt-get install libgoogle-glog-dev libgflags-dev

sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

## CUDA 10.2 install 

[Link](https://developer.nvidia.com/cuda-10.2-download-archive)

## Download OpenCV 4.2.0

cd ~

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip

unzip opencv.zip

unzip opencv_contrib.zip

## Install OpenCV (C++)

cd opencv-4.2.0

mkdir build

cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_C_COMPILER=/usr/bin/gcc-6 \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.2.0/modules \
-D BUILD_EXAMPLES=ON ..

### If your GPU Architeture < 5.3:

-D WITH_CUDNN=OFF \
-D OPENCV_DNN_CUDA=OFF \

make -j4
sudo make install

sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'

sudo ldconfig

# PLC data
http://kos.informatik.uni-osnabrueck.de/3Dscans/
