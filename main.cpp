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
	std::string root_folder = "/kaggle/input/image-super-resolution/dataset/train";
	//	std::string root_folder = "dataset/train";
	auto mydata = CustomDataset(root_folder).map(torch::data::transforms::Stack<>());
	size_t dataset_size = *mydata.size();
	size_t batch_size = 16;
	size_t num_iteration_per_epoch = (dataset_size + batch_size - 1) / batch_size;

	auto data_loader = torch::data::make_data_loader(std::move(mydata), torch::data::DataLoaderOptions().batch_size(batch_size).workers(1));

	Net model;
	torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(1e-4));

	int epochs = 100;
	for (int epoch=0;epoch<epochs;epoch++){
		float epoch_loss = 0;
		for (auto& batch: *data_loader){
			//low_re, high_re = low_re.to(device), high_re.to(device);
			optimizer.zero_grad();
			torch::Tensor pred = model.forward(batch.data);
			torch::Tensor loss = torch::nn::functional::mse_loss(pred, batch.target);
			loss.backward();
			optimizer.step();

			epoch_loss += loss.item<float>();

		}
		std::cout << "Epoch" <<  epoch+1 << " : Loss=" << epoch_loss/(float)num_iteration_per_epoch << '\n';

	}

	return 0;
}
