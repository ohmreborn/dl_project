#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "dataset.h"
#include <iostream>

class NetImpl : public torch::nn::Module {
	private: 
		torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
		torch::nn::ReLU relu{nullptr};

	public: 
		NetImpl(): 
			conv1(torch::nn::Conv2dOptions(3, 64, 9).padding(4)),
			conv2(torch::nn::Conv2dOptions(64, 32, 5).padding(2)),
			conv3(torch::nn::Conv2dOptions(32, 3, 5).padding(2))
	{
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
	}

		torch::Tensor forward(torch::Tensor input) {
			input = torch::relu(conv1(input));
			input = torch::relu(conv2(input));
			input = conv3(input);
			return input;
		}
};
TORCH_MODULE(Net);

int main() {
	std::string root_folder = "/root/.cache/kagglehub/datasets/adityachandrasekhar/image-super-resolution/versions/2/dataset";
	//	std::string root_folder = "dataset";
	std::string train_folder = join_path(root_folder,"train");
	auto train_data = CustomDataset(train_folder).map(torch::data::transforms::Stack<>());
	size_t train_size = *train_data.size();
	size_t batch_size = 16;
	size_t num_train_iteration_per_epoch = (train_size + batch_size - 1) / batch_size;
	auto train_loader = torch::data::make_data_loader(std::move(train_data), torch::data::DataLoaderOptions().batch_size(batch_size).workers(1));

	torch::Device device = torch::cuda::is_available() ? 
		torch::kCUDA : torch::kCPU;
	std::cout << device << '\n';
	Net model;
	model->to(device);
	torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(1e-4));

	std::cout << "start train \n";
	int epochs = 100;
	for (int epoch=0;epoch<epochs;epoch++){
		float epoch_loss = 0;
		for (auto& batch: *train_loader){
			torch::Tensor data = batch.data.to(device);
			torch::Tensor target = batch.target.to(device);
			torch::Tensor pred = model(data);
			torch::Tensor loss = torch::nn::functional::mse_loss(pred, target);
			loss.backward();
			optimizer.step();
			optimizer.zero_grad();
			epoch_loss += loss.item<float>();
		}
		std::cout << "Epoch" <<  epoch+1 << " : Loss=" << epoch_loss/(float)num_train_iteration_per_epoch << '\n';
	}
	torch::save(model, "model.pt");

	return 0;
}

