#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

class Net : public torch::nn::Module {
	private: 
		torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
		torch::nn::ReLU relu{nullptr};

	public: 
		Net(): 
			conv1(torch::nn::Conv2dOptions(3, 64, 9).padding(4)),
			conv2(torch::nn::Conv2dOptions(64, 32, 5).padding(2)),
			conv3(torch::nn::Conv2dOptions(32, 3, 5).padding(2))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
	}

		torch::Tensor forward(torch::Tensor input) {
			input = relu(conv1(input));
			input = relu(conv2(input));
			input = conv3(input);
			return input;
		}
};

int main() {
	std::string filename = "./data/dataset/train/low_res/0.png";
	std::cout << "Before cv::imread\n";
	cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
	std::cout << "After cv::imread\n";
	if (img.empty()){
		std::cerr << "Image not found!";
		return 1;
	}
	long long kRows = img.rows;
	long long kCols = img.cols;
	long long channels = img.channels();
	std::cout << "Before cvtColor\n";
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	std::cout << "After cvtColor\n";

	std::cout << "Before from_blob\n";
	std::cout << "img.isContinuous() = " << img.isContinuous() << '\n';
	std::cout << "img.type() = " << img.type() << " (CV_8UC3 is 16)\n";
	std::cout << "img.channels() = " << img.channels() << '\n';
	std::cout << "img.elemSize() = " << img.elemSize() << '\n';
	std::cout << "img.total() = " << img.total() << '\n';
	std::cout << "img.step[0] = " << img.step[0] << " step[1] = " << img.step[1] << '\n';
	std::cout << "img.data ptr = " << static_cast<const void*>(img.data) << '\n';
	torch::Tensor img_tensor = torch::from_blob(
			img.data,
			{img.rows, img.cols, 3},
			torch::kUInt8
			).clone();
	std::cout << "After from_blob\n";

	std::cout << "Before permute\n";
	img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255);
	std::cout << "After permute\n";
	std::cout << img_tensor << '\n';

	return 0;
}
