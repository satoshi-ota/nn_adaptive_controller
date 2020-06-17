#include <controller_network/architecture.h>

ArchitectureImpl::ArchitectureImpl(int in_features, int out_features)
    : dense1_{torch::nn::Linear{100, 12}},
      dense2_{torch::nn::Linear{100, 100}},
      dense3_{torch::nn::Linear{6, 100}},
      conv1_{nn::Conv2dOptions(3, 100, 1)},
      conv2_{nn::Conv2dOptions(100, 200, 1)},
      dense3_{torch::nn::Linear{200, 6}},

{
    register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
    register_module("dense3_", dense3_);
    register_module("conv1_", conv1_);
    register_module("conv2_", conv2_);
    register_module("dense4_", dense3_);
}

torch::Tensor ArchitectureImpl::forward(torch::Tensor x1, torch::Tensor x2)
{
    x1 = torch::relu(dense1_->forward(x)); // [batch_size, N_HIDDEN1]
    x1 = torch::relu(dense2_->forward(x)); // [batch_size, N_HIDDEN2]
    x1 = torch::relu(dense3_->forward(x)); // [batch_size, N_HIDDEN2]
    x = torch::cat({x1, x2});
    x = relu(conv1_(x));
    x = relu(conv2_(x));
    return dense4_->forward(x);
}