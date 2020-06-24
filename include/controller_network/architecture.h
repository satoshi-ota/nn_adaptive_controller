
#ifndef NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H
#define NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H
#include <torch/torch.h>

class ArchitectureImpl : public torch::nn::Module
{
public:
    ArchitectureImpl(int in_feature, int out_features);
    ArchitectureImpl(const ArchitectureImpl &) = delete;
    ArchitectureImpl &operator=(const ArchitectureImpl &) = delete;

    torch::Tensor forward(torch::Tensor &input);

private:
    torch::nn::Linear dense1_;
    torch::nn::Linear dense2_;
    torch::nn::Linear dense3_;
    torch::nn::Linear dense4_;
    torch::nn::Linear dense5_;
    torch::nn::Conv2d conv1_;
    torch::nn::Conv2d conv2_;
};

TORCH_MODULE(Architecture);

#endif // NN_ADAPTIVE_CONTROLLER_ARCHITECTURE_H