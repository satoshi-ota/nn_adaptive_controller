
#include <controller_network/custom_dataset.h>

CustomDataset::CustomDataset(std::vector<std::ifstream> &ifss)
    : inputs_{},
      outputs_{}
{
    inputs_.reserve(64 * 24);
    outputs_.reserve(64 * 24);

    std::ifstream ifs("data.csv");

    std::string line;

    // for (auto i = 0; i < 64; ++i)
    // {
    //     inputs_.emplace_back(torch::ones({1, 1, 24}));
    //     outputs_.emplace_back(torch::ones({1, 6}));
    // }
    while (getline(ifs, line))
    {
        inputs_.emplace_back(torch::ones({1, 1, 24}));
        outputs_.emplace_back(torch::ones({1, 6}));
    }
};