#include <controller_network/architecture.h>

ArchitectureImpl::ArchitectureImpl(int in_features, int out_features)
    : dense1_{torch::nn::Linear{12, 100}},
      dense2_{torch::nn::Linear{100, 100}},
      dense3_{torch::nn::Linear{100, 6}},
      conv1_{torch::nn::Conv2dOptions(3, 100, 1)},
      conv2_{torch::nn::Conv2dOptions(100, 200, 1)},
      dense4_{torch::nn::Linear{150, 1}},
      //   dense4_{torch::nn::Linear{200 * 1 * 6, 6}},
      //   dense5_{torch::nn::Linear{6, 8}},
      mode_{OFFLINE}

{
    register_module("dense1_", dense1_);
    register_module("dense2_", dense2_);
    register_module("dense3_", dense3_);
    register_module("conv1_", conv1_);
    register_module("conv2_", conv2_);
    register_module("dense4_", dense4_);
    // register_module("dense5_", dense5_);
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
    // std::cout << x.sizes() << '\n';
    x = x.view({x.size(0), 1, 8, 150});
    // x = x.view({x.size(0), 200 * 1 * 6});
    // std::cout << x.sizes() << '\n';

    x = dense4_->forward(x);

    // if (mode_ == ADAPTATION)
    // {
    //     std::cout << "ADAPTATION" << '\n';
    //     auto new_weight = torch::randn({8, 6});
    //     auto new_weight_gpu = new_weight.to(torch::kCUDA);
    //     dense5_->weight = new_weight_gpu;
    // }

    // if (mode_ == BACKPROP)
    // {
    //     std::cout << "BACKPROP" << '\n';
    //     dense5_->weight.set_requires_grad(false);
    //     dense5_->bias.set_requires_grad(false);
    // }

    // x = dense5_->forward(x);

    std::cout << dense4_->weight.sizes() << '\n';

    return x;
}
