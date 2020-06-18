
#include <controller_network/custom_dataset.h>
// #include "dataset_reader.h"

// namespace
// {
//     constexpr int IMAGE_SIZE{100};

//     torch::Tensor convert_to_tensor(const cv::Mat &image)
//     {
//         cv::Mat fimage{};
//         image.convertTo(fimage, CV_32FC3);
//         auto tensor = torch::from_blob(fimage.data, {image.rows, image.cols, 3}, torch::kFloat);
//         tensor = tensor.permute({2, 0, 1});
//         return tensor.clone();
//     }

//     torch::Tensor convert_to_tensor(int label)
//     {
//         auto tensor = torch::empty(1, torch::kInt64);
//         *reinterpret_cast<int64_t *>(tensor.data_ptr()) = label;
//         return tensor;
//     }
// } // namespace

CustomDataset::CustomDataset(std::vector<std::ifstream> &ifss)
    : inputs_{},
      outputs_{}
{
    // auto size = ifss.size();

    inputs_.reserve(64 * 24);
    outputs_.reserve(64 * 24);
    // cv::Mat image;
    // int label;

    // for (auto& ifs : ifss)
    // {
    //     DatasetReader reader{ifs};
    for (auto i = 0; i < 64; ++i)
    {
        //   std::tie(input, output) = reader.load_one_image(i);
        inputs_.emplace_back(torch::ones({1, 1, 24}));
        outputs_.emplace_back(torch::ones({6, 1}));
    }
    // }
};