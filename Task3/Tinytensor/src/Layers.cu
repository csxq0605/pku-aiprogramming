#ifndef LAYERS_INL
#define LAYERS_INL

#include <cublas_v2.h>
#include "Layers_kernels.cuh"
#include "Layers.h"
#include "base.h"

template <typename Type>
void FcForward(
    const TinyTensor<Type>& input,
    TinyTensor<Type>& output,
    const TinyTensor<Type>& weight,
    const TinyTensor<Type>& bias
){
    int batch_size = 1;
    std::vector<int> output_shape;
    for (int i = 0; i < input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
        output_shape.push_back(input.shape[i]);
    }
    int feature_in = input.shape.back();
    int feature_out = weight.shape.back();
    output_shape.push_back(feature_out);
    output.resize(output_shape);
    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, feature_out, feature_in, 1, input.p_data, weight.p_data, 0, output.p_data);
    TinyTensor<Type> Ones = {std::vector<int>{batch_size, 1}, "gpu"};
    Ones.ones();
    if (bias.shape.size() == 1 && bias.shape[0] == 1){
        TinyTensor<Type> Bias = bias;
        double bias_val = Bias.cpu().p_data[0];
        TinyTensor<Type> BiasTensor = {std::vector<int>{1, feature_out}, "gpu"};
        BiasTensor.floatks(bias_val);
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, feature_out, 1, 1, Ones.p_data, BiasTensor.p_data, 1, output.p_data);
    }
    else{
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, feature_out, 1, 1, Ones.p_data, bias.p_data, 1, output.p_data);
    }
}

template <typename Type>
void FcBackward(
    const TinyTensor<Type>& input,
    const TinyTensor<Type>& output,
    const TinyTensor<Type>& weight,
    const TinyTensor<Type>& bias,
    TinyTensor<Type>& grad_input,
    const TinyTensor<Type>& grad_output,
    TinyTensor<Type>& grad_weight,
    TinyTensor<Type>& grad_bias
){
    grad_input.resize(input.shape);
    grad_input.zeros();
    grad_weight.resize(weight.shape);
    grad_weight.zeros();
    grad_bias.resize(bias.shape);
    grad_bias.zeros();

    int batch_size = 1;
    for (int i = 0; i < input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
    }
    int feature_in = input.shape.back();
    int feature_out = weight.shape.back();
    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_T, batch_size, feature_in, feature_out, 1, grad_output.p_data, weight.p_data, 0, grad_input.p_data);
    cudaGemm(CUBLAS_OP_T, CUBLAS_OP_N, feature_in, feature_out, batch_size, 1, input.p_data, grad_output.p_data, 0, grad_weight.p_data);
    TinyTensor<Type> Ones = {std::vector<int>{1, batch_size}, "gpu"};
    Ones.ones();
    if (bias.shape.size() == 1 && bias.shape[0] == 1){
        TinyTensor<Type> BiasTensor = {std::vector<int>{1, feature_out}, "gpu"};
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, feature_out, batch_size, 1, Ones.p_data, grad_output.p_data, 0, BiasTensor.p_data);
        TinyTensor<Type> Biasones = {std::vector<int>{feature_out, 1}, "gpu"};
        Biasones.ones();
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, feature_out, 1, BiasTensor.p_data, Biasones.p_data, 0, grad_bias.p_data);
    }
    else{
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, feature_out, batch_size, 1, Ones.p_data, grad_output.p_data, 0, grad_bias.p_data);
    }
}

template <typename Type>
void im2col(
    const TinyTensor<Type>& im_tensor,
    TinyTensor<Type>& col_tensor,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    int batch_size = 1;
    for (int i = 0; i < im_tensor.shape.size() - 3; i++){
        batch_size *= im_tensor.shape[i];
    }
    int channels = im_tensor.shape[im_tensor.shape.size() - 3];
    int im_h = im_tensor.shape[im_tensor.shape.size() - 2];
    int im_w = im_tensor.shape.back();
    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<int> col_shape;
    if (im_tensor.shape.size() == 3){
        col_shape = std::vector<int>{col_h * col_w, channels * kernel_h * kernel_w};
    }
    else{
        col_shape = std::vector<int>{batch_size, col_h * col_w, channels * kernel_h * kernel_w};
    }
    col_tensor.resize(col_shape);
    int kernels_num = channels * col_h * col_w;
    int im_size = channels * im_h * im_w;
    int col_size = kernels_num * kernel_h * kernel_w;

    for (int i = 0; i < batch_size; i++){
        cudaIm2Col<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(
            im_tensor.p_data + i * im_size, col_tensor.p_data + i * col_size, kernels_num, channels, col_h, col_w, im_h,im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    }
    cudaDeviceSynchronize();
}

template <typename Type>
void col2im(
    const TinyTensor<Type>& col_tensor,
    TinyTensor<Type>& im_tensor,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& im_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    int batch_size = 1;
    for (int i = 0; i < im_shape.size() - 3; i++){
        batch_size *= im_tensor.shape[i];
    }
    int channels = im_shape[im_shape.size() - 3];
    int im_h = im_shape[im_shape.size() - 2];
    int im_w = im_shape.back();
    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<int> in_shape;
    if (im_shape.size() == 3){
        in_shape = std::vector<int>{channels, im_h, im_w};
    }
    else{
        in_shape = std::vector<int>{batch_size, channels, im_h, im_w};
    }
    im_tensor.resize(in_shape);
    int kernels_num = channels * im_h * im_w;
    int im_size = kernels_num;
    int col_size = channels * col_h * col_w * kernel_h * kernel_w;
    for (int i = 0; i < batch_size; i++){
        cudaCol2Im<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(col_tensor.p_data + i * col_size, im_tensor.p_data + i * im_size, kernels_num, channels, col_h, col_w, im_h, im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    }
    cudaDeviceSynchronize();
}

template <typename Type>
void ConvForward(
    const TinyTensor<Type>& input,
    TinyTensor<Type>& output,
    const TinyTensor<Type>& weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    int batch_size = 1;
    std::vector<int> output_shape;
    for (int i = 0; i < input.shape.size() - 3; i++) {
        batch_size *= input.shape[i];
        output_shape.push_back(input.shape[i]);
    }
    int channels_in = input.shape[input.shape.size() - 3];
    int im_h = input.shape[input.shape.size() - 2];
    int im_w = input.shape.back();
    int channels_out = weight.shape[weight.shape.size() - 4];
    int kernel_h = weight.shape[weight.shape.size() - 2];
    int kernel_w = weight.shape.back();
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;
    output_shape.push_back(channels_out);
    output_shape.push_back(col_h);
    output_shape.push_back(col_w);
    output.resize(output_shape);
    TinyTensor<Type> input_resize = input;
    input_resize.resize({batch_size, channels_in, im_h, im_w});

    int output_size = channels_out * col_h * col_w;
    for (int i = 0; i < batch_size; i++){
        TinyTensor<Type> input_col = {std::vector<int>{col_h * col_w, channels_in * kernel_h * kernel_w}, "gpu"};
        input_col.zeros();
        TinyTensor<Type> input_slice = input_resize[i];
        im2col(input_slice, input_col, {kernel_h, kernel_w}, pad_h, pad_w, stride_h, stride_w);
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_T, channels_out, col_h * col_w, channels_in * kernel_h * kernel_w, 1, weight.p_data, input_col.p_data, 0, output.p_data + i * output_size);
    }
    cudaDeviceSynchronize();
}

template <typename Type>
void ConvBackward(
    const TinyTensor<Type>& input,
    const TinyTensor<Type>& output,
    const TinyTensor<Type>& weight,
    TinyTensor<Type>& grad_input,
    const TinyTensor<Type>& grad_output,
    TinyTensor<Type>& grad_weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    grad_input.resize(input.shape);
    grad_input.zeros();
    grad_weight.resize(weight.shape);
    grad_weight.zeros();

    int batch_size = 1;
    for (int i = 0; i < input.shape.size() - 3; i++) {
        batch_size *= input.shape[i];
    }
    int channels_in = input.shape[input.shape.size() - 3];
    int im_h = input.shape[input.shape.size() - 2];
    int im_w = input.shape.back();
    int channels_out = weight.shape[weight.shape.size() - 4];
    int kernel_h = weight.shape[weight.shape.size() - 2];
    int kernel_w = weight.shape.back();
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;

    TinyTensor<Type> input_resize = input;
    input_resize.resize({batch_size, channels_in, im_h, im_w});
    TinyTensor<Type> grad_output_resize = grad_output;
    grad_output_resize.resize({batch_size, channels_out, col_h, col_w});
    TinyTensor<Type> grad_input_col = {std::vector<int>{batch_size, col_h * col_w, channels_in * kernel_h * kernel_w}, "gpu"};
    grad_input_col.zeros();

    int input_col_size = col_h * col_w * channels_in * kernel_h * kernel_w;
    for (int i = 0; i < batch_size; i++){
        TinyTensor<Type> input_col = {std::vector<int>{col_h * col_w, channels_in * kernel_h * kernel_w}, "gpu"};
        input_col.zeros();
        TinyTensor<Type> input_slice = input_resize[i];
        TinyTensor<Type> grad_output_slice = grad_output_resize[i];
        im2col(input_slice, input_col, {kernel_h, kernel_w}, pad_h, pad_w, stride_h, stride_w);
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, channels_out, channels_in * kernel_h * kernel_w, col_h * col_w, 1, grad_output_slice.p_data, input_col.p_data, 1, grad_weight.p_data);
        cudaGemm(CUBLAS_OP_T, CUBLAS_OP_N, col_h * col_w, channels_in * kernel_h * kernel_w, channels_out, 1, grad_output_slice.p_data, weight.p_data, 0, grad_input_col.p_data + i * input_col_size);
    }
    cudaDeviceSynchronize();
    col2im(grad_input_col, grad_input, {kernel_h, kernel_w}, input.shape, pad_h, pad_w, stride_h, stride_w);
}

template <typename Type>
void MaxPoolingForward(
    const TinyTensor<Type>& input,
    TinyTensor<Type>& output,
    TinyTensor<Type>& mask,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    int batch_size = 1;
    std::vector<int> output_shape;
    for (int i = 0; i < input.shape.size() - 3; i++) {
        batch_size *= input.shape[i];
        output_shape.push_back(input.shape[i]);
    }
    int channels = input.shape[input.shape.size() - 3];
    int im_h = input.shape[input.shape.size() - 2];
    int im_w = input.shape.back();
    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;
    output_shape.push_back(channels);
    output_shape.push_back(col_h);
    output_shape.push_back(col_w);
    output.resize(output_shape);
    mask.resize(output_shape);
    int kernels_num = batch_size * channels * col_h * col_w;
    
    cudaMaxPoolingForward<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(input.p_data, output.p_data, mask.p_data, kernels_num, channels, col_h, col_w, im_h, im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    cudaDeviceSynchronize();
}

template <typename Type>
void MaxPoolingBackward(
    const TinyTensor<Type>& grad_output,
    const TinyTensor<Type>& mask,
    TinyTensor<Type>& grad_input,
    const std::vector<int>& input_shape,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    int batch_size = 1;
    std::vector<int> grad_input_shape;
    for (int i = 0; i < grad_output.shape.size() - 3; i++) {
        batch_size *= grad_output.shape[i];
        grad_input_shape.push_back(grad_output.shape[i]);
    }
    int channels = grad_output.shape[grad_output.shape.size() - 3];
    int col_h = grad_output.shape[grad_output.shape.size() - 2];
    int col_w = grad_output.shape.back();
    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();
    int im_h = input_shape[input_shape.size() - 2];
    int im_w = input_shape.back();
    grad_input_shape.push_back(channels);
    grad_input_shape.push_back(im_h);
    grad_input_shape.push_back(im_w);
    grad_input.resize(grad_input_shape);
    grad_input.zeros();
    int kernels_num = batch_size * channels * col_h * col_w;
    cudaMaxPoolingBackward<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(grad_output.p_data, mask.p_data, grad_input.p_data, kernels_num, channels, col_h, col_w, im_h, im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
    cudaDeviceSynchronize();
}

template <typename Type>
void SoftmaxForward(
    const TinyTensor<Type>& input,
    TinyTensor<Type>& output
){
    int batch_size = 1;
    for (int i = 0; i < input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
    }
    output = input;
    int channels = input.shape.back();

    TinyTensor<Type> temp = {std::vector<int>{batch_size}, "gpu"};
    temp.zeros();
    cudaChannelMax<Type>(input.p_data, temp.p_data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaChannelSub<Type><<<CudaGetBlocks(batch_size), BLOCK_SIZE>>>(temp.p_data, output.p_data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaChannelExp<Type><<<CudaGetBlocks(batch_size * channels), BLOCK_SIZE>>>(output.p_data, batch_size * channels);
    cudaDeviceSynchronize();
    temp.zeros();
    cudaChannelSum<Type>(output.p_data, temp.p_data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaChannelDiv<Type><<<CudaGetBlocks(batch_size), BLOCK_SIZE>>>(temp.p_data, output.p_data, batch_size, channels);
    cudaDeviceSynchronize();
}

template <typename Type>
float SoftmaxLoss(
    const TinyTensor<Type>& softmax_output,
    const TinyTensor<int>& labels
){
    int batch_size = 1;
    for (int i = 0; i < softmax_output.shape.size() - 1; i++) {
        batch_size *= softmax_output.shape[i];
    }
    int channels = softmax_output.shape.back();

    TinyTensor<Type> temp = softmax_output;
    temp.cpu();
    TinyTensor<int> real = labels;
    real.cpu();
    float loss = 0;
    for (int i = 0; i < batch_size; i++){
        int max_idx = 0;
        Type max_val = temp.p_data[i * channels];
        for (int j = 1; j < channels; j++){
            if (temp.p_data[i * channels + j] > max_val){
                max_val = temp.p_data[i * channels + j];
                max_idx = j;
            }
        }
        if (max_idx != real.p_data[i]){
            loss += 1;
        }
    }
    return loss / batch_size;
}

template <typename Type>
void CrossEntropyLoss(
    const TinyTensor<Type>& input,
    const TinyTensor<int>& labels,
    TinyTensor<Type>& loss
){
    int batch_size = 1;
    for (int i = 0; i < input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
    }
    int channels = input.shape.back();

    TinyTensor<Type> temp = {std::vector<int>{batch_size}, "gpu"};
    temp.zeros();
    loss.resize(std::vector<int>{1});
    loss.zeros();
    cudaChannelLog<Type><<<CudaGetBlocks(batch_size), BLOCK_SIZE>>>(input.p_data, temp.p_data, labels.p_data, batch_size, channels);
    cudaDeviceSynchronize();
    cudaChannelSum<Type>(temp.p_data, loss.p_data, 1, batch_size);
    cudaDeviceSynchronize();
}

template <typename Type>
void CrossEntropyLossBackward(
    const TinyTensor<Type>& softmax_output,
    const TinyTensor<int>& labels,
    TinyTensor<Type>& grad_input
){
    int batch_size = 1;
    for (int i = 0; i < softmax_output.shape.size() - 1; i++) {
        batch_size *= softmax_output.shape[i];
    }
    int channels = softmax_output.shape.back();

    TinyTensor<Type> real = {softmax_output.shape, "gpu"};
    real.zeros();
    cudaChannelOne<Type><<<CudaGetBlocks(batch_size), BLOCK_SIZE>>>(labels.p_data, real.p_data, batch_size, channels);
    cudaDeviceSynchronize();
    grad_input.resize(softmax_output.shape);
    grad_input = softmax_output - real;
}

template void FcForward<float>(
    const TinyTensor<float>& input,
    TinyTensor<float>& output,
    const TinyTensor<float>& weight,
    const TinyTensor<float>& bias
);
template void FcBackward<float>(
    const TinyTensor<float>& input,
    const TinyTensor<float>& output,
    const TinyTensor<float>& weight,
    const TinyTensor<float>& bias,
    TinyTensor<float>& grad_input,
    const TinyTensor<float>& grad_output,
    TinyTensor<float>& grad_weight,
    TinyTensor<float>& grad_bias
);
template void im2col<float>(
    const TinyTensor<float>& im_tensor,
    TinyTensor<float>& col_tensor,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void col2im<float>(
    const TinyTensor<float>& col_tensor,
    TinyTensor<float>& im_tensor,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& im_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void ConvForward<float>(
    const TinyTensor<float>& input,
    TinyTensor<float>& output,
    const TinyTensor<float>& weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void ConvBackward<float>(
    const TinyTensor<float>& input,
    const TinyTensor<float>& output,
    const TinyTensor<float>& weight,
    TinyTensor<float>& grad_input,
    const TinyTensor<float>& grad_output,
    TinyTensor<float>& grad_weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void MaxPoolingForward<float>(
    const TinyTensor<float>& input,
    TinyTensor<float>& output,
    TinyTensor<float>& mask,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void MaxPoolingBackward<float>(
    const TinyTensor<float>& grad_output,
    const TinyTensor<float>& mask,
    TinyTensor<float>& grad_input,
    const std::vector<int>& input_shape,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);
template void SoftmaxForward<float>(
    const TinyTensor<float>& input,
    TinyTensor<float>& output
);
template float SoftmaxLoss<float>(
    const TinyTensor<float>& softmax_output,
    const TinyTensor<int>& labels
);
template void CrossEntropyLoss<float>(
    const TinyTensor<float>& input,
    const TinyTensor<int>& labels,
    TinyTensor<float>& loss
);
template void CrossEntropyLossBackward<float>(
    const TinyTensor<float>& softmax_output,
    const TinyTensor<int>& labels,
    TinyTensor<float>& grad_input
);

#endif