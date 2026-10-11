#include "base.h"
#include <vector>
#include <iostream>
#include <random>
#include <iomanip>
#include <fstream>
#include "TinyTensor.cuh"
#include "TinyTensor_kernels.cuh"
#include "Layers.cuh"
#include "Layers_kernels.cuh"

#define Type float
#define T TinyTensor<Type>
std::ofstream file1, file2;

std::vector<Type> generateRandomData(size_t size, Type min = -10.0, Type max = 10.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Type> dis(min, max);

    std::vector<Type> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

void data_output(TinyTensor<int> tensor, std::ofstream& file) {
    int* data = new int[tensor.size];
    if (tensor.device == "gpu"){
        cudaMemcpy(data, tensor.p_data, tensor.size * sizeof(int), cudaMemcpyDeviceToHost);
    }
    else{
        data = tensor.p_data;
    }
    for (int i = 0; i < tensor.size; i++){
        file << data[i] << " ";
    }
    file << std::endl;
    delete[] data;
}

void data_output(T tensor, std::ofstream& file) {
    Type* data = new Type[tensor.size];
    if (tensor.device == "gpu"){
        cudaMemcpy(data, tensor.p_data, tensor.size * sizeof(Type), cudaMemcpyDeviceToHost);
    }
    else{
        data = tensor.p_data;
    }
    for (int i = 0; i < tensor.size; i++){
        file << std::fixed << std::setprecision(5) << data[i] << " ";
    }
    file << std::endl;
    delete[] data;
}

int main() {

    file1.open("testdata.txt");
    file2.open("cppout.txt");

    // Test 1
    std::vector<int> input_shape = {2, 3}; // 2 samples, 3 features each
    std::vector<int> weight_shape = {3, 4}; // 3 input features, 4 output features
    std::vector<int> bias_shape = {4}; // 4 output features
    std::vector<int> output_shape = {2, 4}; // 2 samples, 4 output features each

    T input(input_shape, "gpu");
    T weight(weight_shape, "gpu");
    T bias(bias_shape, "gpu");
    T output(output_shape, "gpu");

    input.random(0, 10);
    weight.random(0, 10);
    bias.random(0, 10);

    std::cout << "Input: " << std::endl;
    std::cout << input << std::endl;
    data_output(input, file1);
    std::cout << "Weight: " << std::endl;
    std::cout << weight << std::endl;
    data_output(weight, file1);
    std::cout << "Bias: " << std::endl;
    std::cout << bias << std::endl;
    data_output(bias, file1);

    FcForward(input, output, weight, bias);
    std::cout << "Output: " << std::endl;
    std::cout << output << std::endl;
    data_output(output, file2);
    std::cout << "Test 1 passed" << std::endl;

    // Test 2
    T grad_input(input_shape, "gpu");
    T grad_weight(weight_shape, "gpu"); 
    T grad_bias(bias_shape, "gpu");
    T grad_output(output_shape, "gpu");

    grad_output.random(0, 10);

    std::cout << "Grad Output: " << std::endl;
    std::cout << grad_output << std::endl;
    data_output(grad_output, file1);

    FcBackward(input, output, weight, bias, grad_input, grad_output, grad_weight, grad_bias);
    std::cout << "Grad Input: " << std::endl;
    std::cout << grad_input << std::endl;
    data_output(grad_input, file2);
    std::cout << "Grad Weight: " << std::endl;
    std::cout << grad_weight << std::endl;
    data_output(grad_weight, file2);
    std::cout << "Grad Bias: " << std::endl;
    std::cout << grad_bias << std::endl;
    data_output(grad_bias, file2);
    std::cout << "Test 2 passed" << std::endl;

    // Test 3
    input_shape = {2, 3, 5, 5};
    weight_shape = {6, 3, 3, 3};
    output_shape = {2, 6, 3, 3};

    input = T(input_shape, "gpu");
    weight = T(weight_shape, "gpu");
    output = T(output_shape, "gpu");

    input.random(0, 10);
    weight.random(0, 10);

    std::cout << "Input: " << std::endl;
    std::cout << input << std::endl;
    data_output(input, file1);
    std::cout << "Weight: " << std::endl;
    std::cout << weight << std::endl;
    data_output(weight, file1);

    ConvForward(input, output, weight, 0, 0, 1, 1);
    std::cout << "Output: " << std::endl;
    std::cout << output << std::endl;
    data_output(output, file2);
    std::cout << "Test 3 passed" << std::endl;

    // Test 4
    grad_input = T(input_shape, "gpu");
    grad_weight = T(weight_shape, "gpu");
    grad_output = T(output_shape, "gpu");

    grad_output.random(0, 10);

    std::cout << "Grad Output: " << std::endl;
    std::cout << grad_output << std::endl;
    data_output(grad_output, file1);

    ConvBackward(input, output, weight, grad_input, grad_output, grad_weight, 0, 0, 1, 1);
    std::cout << "Grad Input: " << std::endl;
    std::cout << grad_input << std::endl;
    data_output(grad_input, file2);
    std::cout << "Grad Weight: " << std::endl;
    std::cout << grad_weight << std::endl;
    data_output(grad_weight, file2);
    std::cout << "Test 4 passed" << std::endl;

    //test 5
    input_shape = {2, 3, 5, 5};
    output_shape = {2, 3, 2, 2};

    input = T(input_shape, "gpu");
    output = T(output_shape, "gpu");
    T mask(output_shape, "gpu");

    input.random(0, 10);

    std::cout << "Input: " << std::endl;
    std::cout << input << std::endl;
    data_output(input, file1);

    MaxPoolingForward(input, output, mask, {2, 2}, 0, 0, 2, 2);
    std::cout << "Output: " << std::endl;
    std::cout << output << std::endl;
    data_output(output, file2);
    std::cout << "Mask: " << std::endl;
    std::cout << mask << std::endl;
    data_output(mask, file2);
    std::cout << "Test 5 passed" << std::endl;

    //test 6
    grad_input = T(input_shape, "gpu");
    grad_output = T(output_shape, "gpu");

    grad_output.random(0, 10);

    std::cout << "Grad Output: " << std::endl;
    std::cout << grad_output << std::endl;
    data_output(grad_output, file1);

    MaxPoolingBackward(grad_output, mask, grad_input, input_shape, {2, 2}, 0, 0, 2, 2);
    std::cout << "Grad Input: " << std::endl;
    std::cout << grad_input << std::endl;
    data_output(grad_input, file2);
    std::cout << "Test 6 passed" << std::endl;

    //test 7
    input_shape = {50, 10};
    output_shape = {50, 10};

    input = T(input_shape, "gpu");
    output = T(output_shape, "gpu");

    input.random(0, 10);

    std::cout << "Input: " << std::endl;
    std::cout << input << std::endl;
    data_output(input, file1);

    SoftmaxForward(input, output);
    std::cout << "Output: " << std::endl;
    std::cout << output << std::endl;
    data_output(output, file2);
    std::cout << "Test 7 passed" << std::endl;

    //test 8
    std::vector<int> label_shape = {50};
    TinyTensor<int> label(label_shape, "gpu");

    label.random(0, 10);

    std::cout << "Label: " << std::endl;
    std::cout << label << std::endl;
    data_output(label, file1);

    float loss1 = SoftmaxLoss(output, label);
    std::cout << "Loss: " << std::endl;
    std::cout << loss1 << std::endl;
    file2 << loss1 << std::endl;
    std::cout << "Test 8 passed" << std::endl;

    //test 9
    T loss({1}, "gpu");

    CrossEntropyLoss(output, label, loss);
    std::cout << "Loss: " << std::endl;
    std::cout << loss << std::endl;
    data_output(loss, file2);
    std::cout << "Test 9 passed" << std::endl;

    //test 10
    grad_input = T(input_shape, "gpu");

    CrossEntropyLossBackward(output, label, grad_input);
    std::cout << "Grad Input: " << std::endl;
    std::cout << grad_input << std::endl;
    data_output(grad_input, file2);
    std::cout << "Test 10 passed" << std::endl;

    file1.close();
    file2.close();

    std::cout << "All tests passed, enter q to quit" << std::endl;
    char c;
    std::cin >> c;
    while (c != 'q') {
        std::cin >> c;
    }
    return 0;
}