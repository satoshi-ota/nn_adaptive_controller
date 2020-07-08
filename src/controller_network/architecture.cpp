#include <controller_network/architecture.h>

ArchitectureImpl::ArchitectureImpl(int in_features, int out_features)
    // : dense1_{torch::nn::Linear{4, 16}},
    //   dense2_{torch::nn::Linear{16, 16}},
    //   dense3_{torch::nn::Linear{16, 2}},
    //   dense4_{torch::nn::Linear{100, 10}},
    //   dense5_{torch::nn::Linear{10, 3}},
    : conv1_{torch::nn::Conv2dOptions(4, 64, 1)},
      //   conv2_{torch::nn::Conv2dOptions(64, 128, 1)},
      //   conv3_{torch::nn::Conv2dOptions(128, 256, 1)},
      //   conv3_{torch::nn::Conv2dOptions(200, 400, 1)},
      //   dense4_{torch::nn::Linear{400, 200}},
      // dense4_{torch::nn::Linear{200 * 1 * 6, 6}},
      dense5_{torch::nn::Linear{64, 1}},
      batch_norm1_{torch::nn::BatchNorm2dOptions(64).eps(0.5).momentum(0.5).affine(false).track_running_stats(true)},
      //   batch_norm2_{torch::nn::BatchNorm2dOptions(128).eps(0.5).momentum(0.5).affine(false).track_running_stats(true)},
      //   batch_norm3_{torch::nn::BatchNorm2dOptions(256).eps(0.5).momentum(0.5).affine(false).track_running_stats(true)},
      mode_{OFFLINE}

{
    // register_module("dense1_", dense1_);
    // register_module("dense2_", dense2_);
    // register_module("dense3_", dense3_);
    // register_module("dense4_", dense4_);
    // register_module("dense5_", dense5_);
    register_module("conv1_", conv1_);
    // register_module("conv2_", conv2_);
    // register_module("conv3_", conv3_);
    // register_module("dense4_", dense4_);
    register_module("dense5_", dense5_);
    register_module("batch_norm1", batch_norm1_);
    // register_module("batch_norm2", batch_norm2_);
    // register_module("batch_norm3", batch_norm3_);
    w1.clear();
}

torch::Tensor ArchitectureImpl::forward(torch::Tensor &input)
{
    // auto x = dense1_->forward(input);
    // auto xv = input.split(4, 3);
    // auto x = torch::relu(dense1_->forward(xv[0]));
    // x = torch::relu(dense2_->forward(x));
    // x = torch::tanh(dense3_->forward(x));
    // x = torch::tanh(dense4_->forward(x));
    // x = dense5_->forward(x);

    auto x_split = input.split(2, 3);
    auto x = torch::cat({x_split[0], x_split[1], x_split[2], x_split[3]}, 1);
    // std::cout << x << '\n';
    // x = input.reshape({input.size(0), 4, 1, 2});
    x = torch::leaky_relu(batch_norm1_(conv1_(x)));
    // x = torch::leaky_relu(batch_norm2_(conv2_(x)));
    // x = torch::leaky_relu(batch_norm3_(conv3_(x)));
    x = x.view({x.size(0), 1, 2, 64});
    x = dense5_->forward(x);

    // float w = dense5_->weight[0][0].item<float>();

    w1.push_back(dense5_->weight[0][0].item<float>());

    // auto xv = input.split(6, 3);
    // xv[0] = torch::tanh(batch_norm1_(dense1_->forward(xv[0]))); // [batch_size, N_HIDDEN1]
    // xv[0] = torch::tanh(dense1_->forward(xv[0])); // [batch_size, N_HIDDEN1]
    // xv[0] = dropout(xv[0]);
    // xv[0] = torch::tanh(dense2_->forward(xv[0])); // [batch_size, N_HIDDEN2]
    // xv[0] = dense1_->forward(xv[0]); // [batch_size, N_HIDDEN2]
    // xv[0] = dense2_->forward(xv[0]); // [batch_size, N_HIDDEN2]
    // xv[0] = dense3_->forward(xv[0]); // [batch_size, N_HIDDEN2]
    // xv[0] = torch::tanh(dense3_->forward(xv[0])); // [batch_size, N_HIDDEN2]

    // std::vector<torch::Tensor> xvv = xv[1].split(3, 3);

    // auto x = torch::cat({xv[0], xvv[0], xvv[1]}, 1);

    // x = torch::tanh(conv1_(x));
    // x = torch::tanh(conv2_(x));

    // x = conv3_(x);
    // // std::cout << x.sizes() << '\n';
    // x = x.view({x.size(0), 1, 3, 200});
    // // x = x.view({x.size(0), 200 * 1 * 6});
    // // std::cout << x.sizes() << '\n';

    // x = dense4_->forward(x);

    // x = dense5_->forward(x);

    // std::cout << dense5_->weight[0][0].item<float>() << '\n';

    return x;
}
