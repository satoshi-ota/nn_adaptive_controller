
#ifndef NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H
#define NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H
#include <torch/torch.h>

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

    bool updateOutputLayer(const torch::Tensor &s)
    {
        auto new_weight = dense4_->weight;
        new_weight = new_weight.mv(s);
        // new_weight = ;
        auto new_weight_gpu = new_weight.to(torch::kCUDA);
        dense4_->weight = new_weight_gpu;
    }

    Mode checkMode()
    {
        return mode_;
    }

private:
    torch::nn::Linear dense1_;
    torch::nn::Linear dense2_;
    torch::nn::Linear dense3_;
    torch::nn::Linear dense4_;
    // torch::nn::Linear dense5_;
    torch::nn::Conv2d conv1_;
    torch::nn::Conv2d conv2_;

    Mode mode_;
};

TORCH_MODULE(Architecture);

#endif // NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H