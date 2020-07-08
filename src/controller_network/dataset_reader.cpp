#include <controller_network/dataset_reader.h>
#include <string>
#include <map>
#include <iostream>
#include <sstream>

const float RAD_2_DEG = 180 / M_PI;

DatasetReader::DatasetReader(std::ifstream &ifs)
    : ifs_{ifs} {}

std::pair<torch::Tensor, torch::Tensor> DatasetReader::loadFromCSV(std::string line)
{
    std::vector<std::string> strvec = split(line, ',');
    std::vector<float> in_vec, out_vec;

    in_vec.clear();
    out_vec.clear();

    for (int i = 0; i < 8; i++)
    {
        in_vec.push_back(std::stof(strvec[i]));
    }

    for (int i = 8; i < 10; i++)
    {
        out_vec.push_back(std::stof(strvec[i]));
    }

    input_ = torch::from_blob(in_vec.data(), {1, 1, static_cast<unsigned int>(in_vec.size())}).clone();
    output_ = torch::from_blob(out_vec.data(), {1, static_cast<unsigned int>(out_vec.size())}).clone();

    // std::cout << in_vec << '\n';
    // std::cout << output_ << '\n';

    return {input_, output_};
}

std::vector<std::string> DatasetReader::split(std::string &input, char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter))
    {
        result.push_back(field);
    }
    return result;
}