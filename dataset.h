#pragma once
#include <filesystem>
#include <vector>

class CustomDataset : public torch::data::Dataset<CustomDataset, torch::data::Example<>> {
private:
	std::vector<torch::Tensor> low_res;
	std::vector<torch::Tensor> high_res;

public:
    CustomDataset(const std::string root_folder); 
	void read_image(const std::string image_folder, std::vector<torch::Tensor> &data);

    torch::data::Example<> get(size_t index) override; 

    torch::optional<size_t> size() const override; 
};
