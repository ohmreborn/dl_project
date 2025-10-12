#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "dataset.h"

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
//	std::string root_folder = "/kaggle/input/image-super-resolution";
	std::string root_folder = "dataset/train";
	CustomDataset mydata(root_folder);

	/*
	   Net model;
	   std::cout << model.forward(img_tensor);
	   */

	return 0;
}
