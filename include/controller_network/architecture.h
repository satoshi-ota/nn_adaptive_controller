
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

    void showInfo()
    {
        // Set the size of output image to 1200x780 pixels
        plt::figure_size(1200, 780);
        // Plot line from given x and y data. Color is selected automatically.
        plt::plot(w1);
        // Plot a red dashed line from given x and y data.
        // plt::plot(x, w, "r--");
        // Plot a line whose name will show up as "log(x)" in the legend.
        plt::named_plot("Layer:wight", w1);
        // Set x-axis to interval [0,1000000]
        plt::xlim(0, 1000);
        // Add graph title
        plt::title("weight train");
        // Enable legend.
        plt::legend();
        // Save the image (file format is determined by the extension)
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

    std::vector<float> w1, y, z;
    Mode mode_;
};

TORCH_MODULE(Architecture);

#endif // NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H