#ifndef DATASET_READER
#define DATASET_READER

#include <fstream>
#include <utility>

#include <torch/torch.h>

class DatasetReader
{
public:
    DatasetReader(std::ifstream &ifs);
    DatasetReader(const DatasetReader &) = delete;
    DatasetReader &operator=(const DatasetReader &) = delete;

    std::pair<torch::Tensor, torch::Tensor> loadFromCSV(std::string line);
    std::vector<std::string> split(std::string &input, char delimiter);

private:
    std::ifstream &ifs_;

    torch::Tensor input_;
    torch::Tensor output_;
};
#endif // DATASET_READER