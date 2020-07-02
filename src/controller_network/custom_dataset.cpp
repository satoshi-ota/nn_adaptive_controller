
#include <controller_network/custom_dataset.h>

CustomDataset::CustomDataset(std::vector<std::ifstream> &ifss)
    : inputs_{},
      outputs_{}
{
    inputs_.reserve(256 * 12);
    outputs_.reserve(256 * 3);

    std::ifstream ifs("/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/data.csv");

    std::string line;

    // for (auto i = 0; i < 64; ++i)
    // {
    //     inputs_.emplace_back(torch::ones({1, 1, 24}));
    //     outputs_.emplace_back(torch::ones({1, 6}));
    // }
    DatasetReader reader(ifs);

    while (getline(ifs, line))
    {
        torch::Tensor input, output;
        std::tie(input, output) = reader.loadFromCSV(line);
        inputs_.emplace_back(input);
        outputs_.emplace_back(output);
    }
};