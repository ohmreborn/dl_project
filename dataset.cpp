#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "dataset.h"

#include <vector>

CustomDataset::CustomDataset(const std::filesystem::path root_folder) {
	read_image(root_folder / "low_res", low_res);
	read_image(root_folder / "high_res", high_res);
}

void read_image(const std::filesystem::path image_folder, std::vector<torch::Tensor> &data){
	for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
		// Check if the entry is a regular file
		if (std::filesystem::is_regular_file(entry.status())) {
			std::string filename = entry.path().filename();
			cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
			if (img.empty()){
				std::cerr << "Image not found!";
				continue;
			}
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

			torch::Tensor img_tensor = torch::from_blob(
					img.data,
					{img.rows, img.cols, 3},
					torch::kUInt8
					).clone();
			img_tensor = img_tensor.permute({2, 0, 1});
			img_tensor = img_tensor.to(torch::kFloat32).div(255.0);
			data.push_back(img_tensor);
		}
	}
}

torch::data::Example<> CustomDataset::get(size_t index) {
	return {low_res[index], high_res[index]};
}

torch::optional<size_t> CustomDataset::size() const {
	return low_res.size();
}
