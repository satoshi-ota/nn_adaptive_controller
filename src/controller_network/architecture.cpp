#include <controller_network/architecture.h>

ArchitectureImpl::ArchitectureImpl(int in_features, int out_features)
    : dense1_{torch::nn::Linear{12, 100}},
      dense2_{torch::nn::Linear{100, 100}},
      dense3_{torch::nn::Linear{100, 6}},
      conv1_{torch::nn::Conv2dOptions(3, 100, 1)},
      conv2_{torch::nn::Conv2dOptions(100, 200, 1)},
      dense4_{torch::nn::Linear{200, 8}}

{
    register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
    register_module("dense3_", dense3_);
    register_module("conv1_", conv1_);
    register_module("conv2_", conv2_);
    register_module("dense4_", dense4_);
}

torch::Tensor ArchitectureImpl::forward(torch::Tensor x1, torch::Tensor x2)
{
    x1 = torch::relu(dense1_->forward(x1)); // [batch_size, N_HIDDEN1]
    x1 = torch::relu(dense2_->forward(x1)); // [batch_size, N_HIDDEN2]
    x1 = torch::relu(dense3_->forward(x1)); // [batch_size, N_HIDDEN2]
    x1 = torch::cat({x1, x2});
    x1 = torch::relu(conv1_(x1));
    x1 = torch::relu(conv2_(x1));
    return dense4_->forward(x1);
}
