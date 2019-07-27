#include <torch/torch.h>
#include <iostream>
#include <torch/script.h> // One-stop header.

#define USE_OPENCV
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

static std::string classes_label[] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
/**
 * GroundTruth:    cat
 * Predicted:    dog
 * Predicted:  97.75
 */

int main() {
#ifdef WIN32
	std::string mpath = "C:\\SandyWork\\chongqing_work\\Human-Segmentation-PyTorch\\UNet_MobileNetV2.tar\\UNet_MobileNetV2\\";
	std::string model_path = mpath + "UNet_MobileNetV2.pth";
	model_path = "C:\\SandyWork\\mygithub\\pytorch_learn\\train_cnn_cifar10\\output\\1_12000_loss_1.2831.pts";
	std::string image_path = "../cat.jpg";
#else
	std::string model_path = "../../train_cnn_cifar10/output/1_12000_loss_1.2715.pts";
	//model_path = "../../train_cnn_cifar10/output/1_12000_loss_1.2879.pts";
	std::string image_path = "../../train_cnn_cifar10/bb.bmp";
	image_path = "../../train_cnn_cifar10/ttt.bmp";
#endif

#ifdef USE_OPENCV
	cv::Mat image = cv::imread(image_path);
	if(image.empty()){
		std::cout << "Can't imread: " << image_path << std::endl;
		return 0;
	}
	cv::Mat rsz;
	cv::resize(image, rsz, cv::Size(32, 32));
	// cv::namedWindow("test", 1);
	// cv::imshow("test", rsz);
	// cv::waitKey(0);
#endif

	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	} else {
		std::cout << "CUDA is not available!" << std::endl;
	}

	std::shared_ptr<torch::jit::script::Module> module;

	// Deserialize the ScriptModule from a file using torch::jit::load().
	module = torch::jit::load(model_path);
	assert(module != nullptr);

	if (device == torch::kCUDA) {
		module->to(torch::kCUDA);
	}

	std::vector<torch::jit::IValue> inputs;

#ifdef USE_OPENCV
	at::Tensor tensor_image = torch::from_blob(rsz.data, {1, rsz.rows, rsz.cols, 3}, at::kByte);
	tensor_image = tensor_image.permute( { 0, 3, 1, 2 });
	tensor_image = tensor_image.to(at::kFloat);
#else
	// float *pppp = new float[32*32*3*10240];
	// at::Tensor tensor_image = torch::from_blob(pppp, { 10240, 3, 32, 32 }, torch::kFloat32).clone();
	at::Tensor tensor_image = torch::from_blob(g_img_buf, { 1, 3, 32, 32 }, torch::kFloat32).clone();
	//tensor_image = tensor_image.permute( { 0, 3, 1, 2 });
#endif
	
	if (device == torch::kCUDA) {
		tensor_image = tensor_image.to(torch::kCUDA);
	}

	inputs.push_back(tensor_image);
	// Execute the model and turn its output into a tensor.
	std::cout << "start infer" << std::endl;
	auto t1 = std::chrono::high_resolution_clock::now();
	at::Tensor output = module->forward(inputs).toTensor();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> diff = t2 - t1;
	std::cout << "Infer time = " << diff.count() << "ms" << std::endl;

	if (device == torch::kCUDA) {
		output = output.to(at::kCPU);	//inference
	}

//	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

	int output_size = 1;
	for (int i = 0; i < output.sizes().size(); i++) {
		output_size *= output.sizes()[i];
	}

	float maxv = -999999;
	int max_label_id = 0;
	for (int i = 0; i < output_size; i++) {
		if (output.data<float>()[i] > maxv) {
			maxv = output.data<float>()[i];
			max_label_id = i;
		}
	}

	std::cout << "Please test your pickle model by pytorch firstly" << std::endl;
	std::cout << "==============================" << std::endl;
	std::cout << "GroundTruth: cat" << std::endl;
	std::cout << "Predicted: " << classes_label[max_label_id] << std::endl;
	std::cout << "Predicted: " << maxv << std::endl;
	std::cout << "==============================" << std::endl;
	return 0;
}
