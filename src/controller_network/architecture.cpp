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
      dense5_{torch::nn::LinearOptions(64, 1).bias(false)},
      batch_norm1_{torch::nn::BatchNorm2dOptions(64).eps(0.01).momentum(0.1).affine(true).track_running_stats(true)},
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
    w2.clear();
    w3.clear();
    w4.clear();
    w5.clear();
    w6.clear();
    w7.clear();
    w8.clear();
    w9.clear();
    w10.clear();
     c1.clear();
    c2.clear();
    c3.clear();
    c4.clear();
    c5.clear();
    c6.clear();
    c7.clear();
    c8.clear();
    c9.clear();
    c10.clear();
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
    x = torch::relu(batch_norm1_(conv1_(x)));
    // x = torch::leaky_relu(batch_norm2_(conv2_(x)));
    // x = torch::leaky_relu(batch_norm3_(conv3_(x)));
    x = x.view({x.size(0), 1, 2, 64});
    x = dense5_->forward(x);

    c1.push_back(conv1_->weight[0][0].item<float>());
    c2.push_back(conv1_->weight[5][0].item<float>());
    c3.push_back(conv1_->weight[10][0].item<float>());
    c4.push_back(conv1_->weight[15][0].item<float>());
    c5.push_back(conv1_->weight[20][0].item<float>());
    c6.push_back(conv1_->weight[25][0].item<float>());
    c7.push_back(conv1_->weight[30][0].item<float>());
    c8.push_back(conv1_->weight[45][0].item<float>());
    c9.push_back(conv1_->weight[50][0].item<float>());
    c10.push_back(conv1_->weight[55][0].item<float>());

    w1.push_back(dense5_->weight[0][0].item<float>());
    w2.push_back(dense5_->weight[0][5].item<float>());
    w3.push_back(dense5_->weight[0][10].item<float>());
    w4.push_back(dense5_->weight[0][15].item<float>());
    w5.push_back(dense5_->weight[0][20].item<float>());
    w6.push_back(dense5_->weight[0][25].item<float>());
    w7.push_back(dense5_->weight[0][30].item<float>());
    w8.push_back(dense5_->weight[0][35].item<float>());
    w9.push_back(dense5_->weight[0][40].item<float>());
    w10.push_back(dense5_->weight[0][50].item<float>());

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

    // std::cout << conv1_->weight[0][0].item<float>() << '\n';

    return x;
}
