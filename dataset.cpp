#include <torch/torch.h>

class CustomDataset : public torch::data::Dataset<CustomDataset, torch::data::Example<>> {
private:
	std::string root_folder;

public:
    CustomDataset(const std::string& annotations_file, const std::string& img_dir) {
        // Load image paths and labels from annotations_file
        // Populate image_paths and labels vectors
        // Example: Read from a CSV or a manifest file
    }

    torch::data::Example<> get(size_t index) override {
        // Load image from image_paths[index] using OpenCV or similar
        // Convert image to torch::Tensor (e.g., using torch::from_blob)
        // Apply any necessary transformations (resize, normalize)
        torch::Tensor data = ...;

        // Get label from labels[index]
        torch::Tensor target = torch::tensor(labels[index], torch::kInt64);

        return {data, target};
    }

    c10::optional<size_t> size() const override {
        return image_paths.size();
    }
};
