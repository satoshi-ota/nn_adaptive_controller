#include <controller_network/architecture.h>

ArchitectureImpl::ArchitectureImpl(int in_features, int out_features)
    : dense1_{torch::nn::Linear{12, 100}},
      dense2_{torch::nn::Linear{100, 100}},
      dense3_{torch::nn::Linear{100, 6}},
      conv1_{torch::nn::Conv2dOptions(3, 100, 1)},
      conv2_{torch::nn::Conv2dOptions(100, 200, 1)},
      dense4_{torch::nn::Linear{200 * 1 * 6, 6}}

{
    register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
    register_module("dense3_", dense3_);
    register_module("conv1_", conv1_);
    register_module("conv2_", conv2_);
    register_module("dense4_", dense4_);
}

torch::Tensor ArchitectureImpl::forward(torch::Tensor &input)
{
    auto xv = input.split(12, 3);

    xv[0] = torch::relu(dense1_->forward(xv[0])); // [batch_size, N_HIDDEN1]
    xv[0] = torch::relu(dense2_->forward(xv[0])); // [batch_size, N_HIDDEN2]
    xv[0] = torch::relu(dense3_->forward(xv[0])); // [batch_size, N_HIDDEN2]

    std::vector<torch::Tensor> xvv = xv[1].split(6, 3);

    auto x = torch::cat({xv[0], xvv[0], xvv[1]}, 1);

    x = torch::relu(conv1_(x));
    x = conv2_(x);
    std::cout << x.sizes() << '\n';
    x = x.view({x.size(0), 200 * 1 * 6});
    x = dense4_->forward(x);

    return x;
}
