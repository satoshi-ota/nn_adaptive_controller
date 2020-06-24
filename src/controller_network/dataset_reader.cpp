#include <controller_network/dataset_reader.h>
#include <string>
#include <map>
#include <iostream>
#include <sstream>

DatasetReader::DatasetReader(std::ifstream &ifs)
    : ifs_{ifs} {}

std::pair<torch::Tensor, torch::Tensor> DatasetReader::loadFromCSV(std::string line)
{
    std::vector<std::string> strvec = split(line, ',');
    std::vector<float> in_vec, out_vec;

    in_vec.clear();
    out_vec.clear();

    for (int i = 0; i < 24; i++)
    {
        in_vec.push_back(std::stoi(strvec[i]));
    }

    for (int i = 24; i < 32; i++)
    {
        out_vec.push_back(std::stoi(strvec[i]));
    }

    input_ = torch::from_blob(in_vec.data(), {1, 1, static_cast<unsigned int>(in_vec.size())}).clone();
    output_ = torch::from_blob(out_vec.data(), {1, static_cast<unsigned int>(out_vec.size())}).clone();

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