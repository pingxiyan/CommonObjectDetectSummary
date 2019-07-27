# README
test inference based on pytorch cpp interface.

# Requirement
1. Need to download "libtorch" to inference for cpp <br>
	Use pre-build libtorch. 	<br>
	From source code build libtorch 	<br>
2. cudnn: libcudnn7-dev_7.5.1.10-1+cuda9.2_amd64.deb
	$ sudo dpkg -i libcudnn7-dev_7.5.1.10-1+cuda9.2_amd64.deb
3. OpenCV: Image read and show test.

# Convert pickle model to libtorch script model.

	$ train_cnn_cifar10/cvt_model2torchscript.py # refer this script

# Test inference based on libtorch
Inference an image, get classification result.

	mkdir build
	cd build
	cmake -DOpenCV_DIR=[path] -DCMAKE_PREFIX_PATH=[path] -DCUDANN_ROOT_DIR=[your cudnn path] ..
	make -j8
	./testapp

	eg:
	cmake -DCMAKE_PREFIX_PATH=/usr/local/cuda-9.2 -DCUDANN_ROOT_DIR=/usr/lib/x86_64-linux-gnu/ ..


**Result:**

	GroundTruth: cat
	Predicted: dog
	Predicted: 97.7508
	
#### Issues

| Environments | ISSUES |
| -------------------------------------      | ------------------------ |
| pre-build libtorch 1.1 cuda 92 linux       | conflict with OpenCV 4.0 |
| pre-build libtorch 1.1 CPU linux           | conflict with OpenCV 4.0 |
| pre-build libtorch 1.1 cuda 92 Win10       | cuda_is_available() alway return false |
| pre-build libtorch 1.0 cuda 92 Win10 debug | work, but speed is low |
| pre-build libtorch 1.1 CPU Win10           | ok |


**Note** After copying CUDNN to /usr/local/cuda/, don't need to set '-DCUDNN_ROOT_DIR'.

	$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
	$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
	Refer:
	https://github.com/zccyman/pytorch-inference/blob/master/pytorch/src/pytorch_interface.cpp
	https://oldpan.me/archives/pytorch-c-libtorch-inference

# Build pytorch for source code.

	$ git clone --recursive https://github.com/pytorch/pytorch
	$ git submodule sync
	$ git submodule update --init --recursive
	$ cd pytorch

	# build pythorch
	$ python3 setup.py develop

	# build libtorch
	$ mkdir build && cd build
	$ python3 ../tools/build_libtorch.py  # wait for a long time

	Set -DCMAKE_INSTALL_PREFIX=[pytorch]/torch/share/cmake

# Known issues
If we use ourself builded OpenCV, don't known why can't link opencv libraries. Errors log as follow: <br>

	$ cmake -DOpenCV_DIR=/home/xiping/opensource/opencv/build ..
	main.cpp:(.text+0xfe1): undefined reference to `cv::imread(std::string const&, int)'
	main.cpp:(.text+0x10bb): undefined reference to `cv::namedWindow(std::string const&, int)'
	main.cpp:(.text+0x1132): undefined reference to `cv::imshow(std::string const&, cv::_InputArray const&)'

Using this /opt/anaconda/anaconda3/share/OpenCV, can normornly link.  <br>

	$ cmake -DOpenCV_DIR=/opt/anaconda/anaconda3/share/OpenCV ..
	For anaconda OpenCV, don't support imshow.

Using libtorch which builded from source code can fix the issue.

