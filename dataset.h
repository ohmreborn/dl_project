#pragma once
#include <filesystem>


class CustomDataset : public torch::data::Dataset<CustomDataset, torch::data::Example<>> {
private:
	std::vector<torch::Tensor> low_res;
	std::vector<torch::Tensor> high_res;

public:
    CustomDataset(const std::filesystem::path root_folder); 
	void read_image(const std::filesystem::path root_folder, const std::filesystem::path low_high, std::vector<torch::Tensor> &data);

    torch::data::Example<> get(size_t index) override; 

    torch::optional<size_t> size() const override; 
};
