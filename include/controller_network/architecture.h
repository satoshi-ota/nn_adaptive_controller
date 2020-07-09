
#ifndef NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H
#define NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H
#include <torch/torch.h>
#include <controller_network/matplotlibcpp.h>

namespace plt = matplotlibcpp;

enum Mode
{
    OFFLINE,
    BACKPROP,
    ADAPTATION
};

class ArchitectureImpl : public torch::nn::Module
{
public:
    ArchitectureImpl(int in_feature, int out_features);
    ArchitectureImpl(const ArchitectureImpl &) = delete;
    ArchitectureImpl &operator=(const ArchitectureImpl &) = delete;

    torch::Tensor forward(torch::Tensor &input);

    void updateMode(Mode mode)
    {
        mode_ = mode;
    }

    // bool updateOutputLayer(const torch::Tensor &s)
    // {
    //     auto new_weight = dense4_->weight;
    //     new_weight = new_weight.mv(s);
    //     // new_weight = ;
    //     auto new_weight_gpu = new_weight.to(torch::kCUDA);
    //     dense4_->weight = new_weight_gpu;
    // }

    Mode checkMode()
    {
        return mode_;
    }

    void plotLoss(float& loss)
    {
        loss_.push_back(loss);
    }

    void showInfo()
    {
        plt::figure_size(1200, 780);

        plt::subplot(3, 1, 1);
        plt::plot(loss_);
        plt::named_plot("Loss", loss_);
        // plt::xlim(0, 500);

        plt::subplot(3, 1, 2);
        plt::plot(c1);
        plt::plot(c2);
        plt::plot(c3);
        plt::plot(c4);
        plt::plot(c5);
        plt::plot(c6);
        plt::plot(c7);
        plt::plot(c8);
        plt::plot(c9);
        plt::plot(c10);
        plt::named_plot("Layer1:wight", c1);
        // plt::xlim(0, 500);

        plt::subplot(3, 1, 3);
        plt::plot(w1);
        plt::plot(w2);
        plt::plot(w3);
        plt::plot(w4);
        plt::plot(w5);
        plt::plot(w6);
        plt::plot(w7);
        plt::plot(w8);
        plt::plot(w9);
        plt::plot(w10);
        plt::named_plot("Layer:wight", w1);
        // plt::xlim(0, 500);

        plt::title("weight train");
        plt::legend();
        plt::save("/root/catkin_ws/src/nn_adaptive_controller/src/controller_network/basic.png");
    }

private:
    // torch::nn::Linear dense1_;
    // torch::nn::Linear dense2_;
    // torch::nn::Linear dense3_;
    // torch::nn::Linear dense4_;
    torch::nn::Linear dense5_;
    torch::nn::Conv2d conv1_;
    // torch::nn::Conv2d conv2_;
    // torch::nn::Conv2d conv3_;

    torch::nn::BatchNorm2d batch_norm1_;
    // torch::nn::BatchNorm2d batch_norm2_;
    // torch::nn::BatchNorm2d batch_norm3_;

    // torch::nn::Dropout dropout;

    std::vector<float> loss_;
    std::vector<float> c1, c2, c3,c4,c5,c6,c7,c8,c9,c10;
    std::vector<float> w1, w2, w3,w4,w5,w6,w7,w8,w9,w10;
    Mode mode_;
};

TORCH_MODULE(Architecture);

#endif // NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H