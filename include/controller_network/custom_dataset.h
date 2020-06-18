#ifndef CUSTOM_DATASET
#define CUSTOM_DATASET
#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include <controller_network/dataset_reader.h>

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
	// Declare 2 vectors of tensors for inputs{x, xdot, xddot} and outputs{u_cmd}
	std::vector<torch::Tensor> inputs_;
	std::vector<torch::Tensor> outputs_;

public:
	// Constructor
	CustomDataset(std::vector<std::ifstream> &ifs);

	// Override get() function to return tensor at location index
	torch::data::Example<> get(std::size_t index) override
	{
		torch::Tensor sample_in = inputs_.at(index);
		torch::Tensor sample_out = outputs_.at(index);
		return {sample_in.clone(), sample_out.clone()};
	};

	// Return the length of data
	torch::optional<std::size_t> size() const override
	{
		return outputs_.size();
	};
};
#endif // CUSTOM_DATASET