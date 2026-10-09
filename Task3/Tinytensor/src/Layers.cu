#ifndef LAYERS_INL
#define LAYERS_INL

#include <cublas_v2.h>
#include "Layers_kernels.cuh"
#include "Layers.h"
#include "base.h"

// FC 的输入通常是 [..., in]；如果最后一维和权重对不上、但 [N, ...] 展平后正好是 [N, in]，
// 就把它当作 [N, in]（例如卷积输出直接接 FC），省掉一次 reshape 拷贝
template <typename Type>
static void FcDims(const TinyTensor<Type>& input, const TinyTensor<Type>& weight, int& batch_size, int& feature_in,
                   std::vector<int>& output_shape){
    int weight_in = weight.shape[0];
    output_shape.clear();
    if (input.shape.back() != weight_in && input.shape.size() > 2 && input.size == (size_t)input.shape[0] * weight_in){
        batch_size = input.shape[0];
        output_shape.push_back(batch_size);
    }
    else{
        if (input.shape.back() != weight_in){
            throw std::invalid_argument("FC: input features do not match weight");
        }
        batch_size = 1;
        for (int i = 0; i < (int)input.shape.size() - 1; i++) {
            batch_size *= input.shape[i];
            output_shape.push_back(input.shape[i]);
        }
    }
    feature_in = weight_in;
    output_shape.push_back(weight.shape.back());
}

template <typename Type>
void FcForward(
    const TinyTensor<Type>& input,
    TinyTensor<Type>& output,
    const TinyTensor<Type>& weight,
    const TinyTensor<Type>& bias
){
    int batch_size, feature_in;
    std::vector<int> output_shape;
    FcDims(input, weight, batch_size, feature_in, output_shape);
    int feature_out = weight.shape.back();
    output.resize_discard(output_shape);
    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, feature_out, feature_in, 1, input.p_data, weight.p_data, 0, output.p_data);
    cudaAddBias<Type><<<CudaGetBlocks(batch_size * feature_out), BLOCK_SIZE>>>(output.p_data, bias.p_data, batch_size, feature_out, (int)bias.size);
}

// FC + ReLU：bias 和 ReLU 在同一个 kernel 里完成
void FcReluForward(
    const TinyTensor<float>& input,
    TinyTensor<float>& output,
    const TinyTensor<float>& weight,
    const TinyTensor<float>& bias
){
    int batch_size, feature_in;
    std::vector<int> output_shape;
    FcDims(input, weight, batch_size, feature_in, output_shape);
    int feature_out = weight.shape.back();
    output.resize_discard(output_shape);
    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, batch_size, feature_out, feature_in, 1, input.p_data, weight.p_data, 0, output.p_data);
    cudaAddBiasRelu<float><<<CudaGetBlocks(batch_size * feature_out), BLOCK_SIZE>>>(output.p_data, bias.p_data, batch_size, feature_out);
}

// FC + ReLU 的反向：先用一个 kernel 同时算出 ReLU 之后的梯度和 bias 梯度，再做两次 GEMM
void FcReluBackward(
    const TinyTensor<float>& input,
    const TinyTensor<float>& output,
    const TinyTensor<float>& weight,
    TinyTensor<float>& grad_input,
    const TinyTensor<float>& grad_output,
    TinyTensor<float>& grad_weight,
    TinyTensor<float>& grad_bias,
    bool need_grad_input
){
    int batch_size, feature_in;
    std::vector<int> output_shape;
    FcDims(input, weight, batch_size, feature_in, output_shape);
    int feature_out = weight.shape.back();
    grad_weight.resize_discard(weight.shape);
    grad_bias.resize_discard(std::vector<int>{feature_out});
    TinyTensor<float> masked{std::vector<int>{batch_size, feature_out}, "gpu"};
    cudaReluMaskColumnSum<float><<<(feature_out + 31) / 32, dim3(32, 8)>>>(grad_output.p_data, output.p_data, masked.p_data,
                                                                             grad_bias.p_data, batch_size, feature_out);
    if (need_grad_input){
        grad_input.resize_discard(input.shape);
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_T, batch_size, feature_in, feature_out, 1, masked.p_data, weight.p_data, 0, grad_input.p_data);
    }
    cudaGemm(CUBLAS_OP_T, CUBLAS_OP_N, feature_in, feature_out, batch_size, 1, input.p_data, masked.p_data, 0, grad_weight.p_data);
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
    TinyTensor<Type>& grad_bias,
    bool need_grad_input
){
    if (need_grad_input){
        grad_input.resize_discard(input.shape);
    }
    grad_weight.resize_discard(weight.shape);
    grad_bias.resize_discard(bias.shape);

    int batch_size, feature_in;
    std::vector<int> output_shape;
    FcDims(input, weight, batch_size, feature_in, output_shape);
    int feature_out = weight.shape.back();
    // dX = dY * W^T, dW = X^T * dY
    if (need_grad_input){
        cudaGemm(CUBLAS_OP_N, CUBLAS_OP_T, batch_size, feature_in, feature_out, 1, grad_output.p_data, weight.p_data, 0, grad_input.p_data);
    }
    cudaGemm(CUBLAS_OP_T, CUBLAS_OP_N, feature_in, feature_out, batch_size, 1, input.p_data, grad_output.p_data, 0, grad_weight.p_data);
    if (bias.size == 1){
        // 标量 bias：对所有元素求和
        cudaSumToScalar<Type><<<1, BLOCK_SIZE>>>(grad_output.p_data, grad_bias.p_data, batch_size * feature_out, Type(1));
    }
    else{
        cudaColumnSum<Type><<<(feature_out + 31) / 32, dim3(32, 8)>>>(grad_output.p_data, grad_bias.p_data, batch_size, feature_out);
    }
}

// 把 [N, C, H, W]（或 [C, H, W]）展开为每个样本 [col_h * col_w, C * kh * kw] 的列矩阵
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
    for (int i = 0; i < (int)im_tensor.shape.size() - 3; i++){
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
    col_tensor.resize_discard(col_shape);
    int kernels_num = batch_size * channels * col_h * col_w;
    cudaIm2Col<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(
        im_tensor.p_data, col_tensor.p_data, kernels_num, channels, col_h, col_w, im_h, im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
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
    for (int i = 0; i < (int)im_shape.size() - 3; i++){
        batch_size *= im_shape[i];
    }
    int channels = im_shape[im_shape.size() - 3];
    int im_h = im_shape[im_shape.size() - 2];
    int im_w = im_shape.back();
    int kernel_h = kernel_shape[kernel_shape.size() - 2];
    int kernel_w = kernel_shape.back();
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;
    im_tensor.resize_discard(im_shape);
    int kernels_num = batch_size * channels * im_h * im_w;
    cudaCol2Im<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(
        col_tensor.p_data, im_tensor.p_data, kernels_num, channels, col_h, col_w, im_h, im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
}

// 每个输出点的乘加次数 (Cin * kh * kw) 较小时，直接卷积比 im2col + GEMM 更快，也不需要额外显存
inline bool UseDirectConv(int k){
    return k <= 256;
}

// 通道数较小时用直接卷积；否则整个 batch 一次 im2col，再用一次批量 GEMM：out[n] = W * col[n]^T
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
    for (int i = 0; i < (int)input.shape.size() - 3; i++) {
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
    output.resize_discard(output_shape);

    int hw = col_h * col_w;
    int k = channels_in * kernel_h * kernel_w;
    if (UseDirectConv(k)){
        ConvShape s{batch_size, channels_in, channels_out, im_h, im_w, col_h, col_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w};
        ConvDirectForward(input.p_data, weight.p_data, output.p_data, s);
        return;
    }
    TinyTensor<Type> input_col{std::vector<int>{batch_size, hw, k}, "gpu"};
    im2col(input, input_col, {kernel_h, kernel_w}, pad_h, pad_w, stride_h, stride_w);
    cudaGemmStridedBatched(CUBLAS_OP_N, CUBLAS_OP_T, channels_out, hw, k, 1,
        weight.p_data, 0, input_col.p_data, (long long)hw * k, 0,
        output.p_data, (long long)channels_out * hw, batch_size);
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
    const int stride_w,
    bool need_grad_input
){
    if (need_grad_input){
        grad_input.resize_discard(input.shape);
    }
    grad_weight.resize_discard(weight.shape);

    int batch_size = 1;
    for (int i = 0; i < (int)input.shape.size() - 3; i++) {
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
    int hw = col_h * col_w;
    int k = channels_in * kernel_h * kernel_w;

    if (UseDirectConv(k)){
        ConvShape s{batch_size, channels_in, channels_out, im_h, im_w, col_h, col_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w};
        ConvDirectBackwardWeight(input.p_data, grad_output.p_data, grad_weight, s);
        if (need_grad_input){
            ConvDirectBackwardInput(grad_output.p_data, weight.p_data, grad_input.p_data, s);
        }
        return;
    }

    TinyTensor<Type> input_col{std::vector<int>{batch_size, hw, k}, "gpu"};
    im2col(input, input_col, {kernel_h, kernel_w}, pad_h, pad_w, stride_h, stride_w);

    // dW = sum_n dY[n] * col[n]：先批量算出每个样本的 dW，再乘全 1 向量求和
    TinyTensor<Type> grad_weight_batch{std::vector<int>{batch_size, channels_out * k}, "gpu"};
    cudaGemmStridedBatched(CUBLAS_OP_N, CUBLAS_OP_N, channels_out, k, hw, 1,
        grad_output.p_data, (long long)channels_out * hw, input_col.p_data, (long long)hw * k, 0,
        grad_weight_batch.p_data, (long long)channels_out * k, batch_size);
    TinyTensor<Type> ones{std::vector<int>{1, batch_size}, "gpu"};
    ones.ones();
    cudaGemm(CUBLAS_OP_N, CUBLAS_OP_N, 1, channels_out * k, batch_size, 1, ones.p_data, grad_weight_batch.p_data, 0, grad_weight.p_data);

    if (!need_grad_input){
        return;
    }
    // dcol[n] = dY[n]^T * W，再 col2im 回到输入形状
    TinyTensor<Type> grad_input_col{std::vector<int>{batch_size, hw, k}, "gpu"};
    cudaGemmStridedBatched(CUBLAS_OP_T, CUBLAS_OP_N, hw, k, channels_out, 1,
        grad_output.p_data, (long long)channels_out * hw, weight.p_data, 0, 0,
        grad_input_col.p_data, (long long)hw * k, batch_size);
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
    for (int i = 0; i < (int)input.shape.size() - 3; i++) {
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
    output.resize_discard(output_shape);
    mask.resize_discard(output_shape);
    int kernels_num = batch_size * channels * col_h * col_w;

    cudaMaxPoolingForward<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(input.p_data, output.p_data, mask.p_data, kernels_num, channels, col_h, col_w, im_h, im_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
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
    for (int i = 0; i < (int)grad_output.shape.size() - 3; i++) {
        batch_size *= grad_output.shape[i];
        grad_input_shape.push_back(grad_output.shape[i]);
    }
    int channels = grad_output.shape[grad_output.shape.size() - 3];
    int col_h = grad_output.shape[grad_output.shape.size() - 2];
    int col_w = grad_output.shape.back();
    int im_h = input_shape[input_shape.size() - 2];
    int im_w = input_shape.back();
    grad_input_shape.push_back(channels);
    grad_input_shape.push_back(im_h);
    grad_input_shape.push_back(im_w);
    grad_input.resize_discard(grad_input_shape);
    grad_input.zeros();
    int kernels_num = batch_size * channels * col_h * col_w;
    cudaMaxPoolingBackward<Type><<<CudaGetBlocks(kernels_num), BLOCK_SIZE>>>(grad_output.p_data, mask.p_data, grad_input.p_data, kernels_num, channels, col_h, col_w, im_h, im_w);
}

template <typename Type>
void SoftmaxForward(
    const TinyTensor<Type>& input,
    TinyTensor<Type>& output
){
    int batch_size = 1;
    for (int i = 0; i < (int)input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
    }
    int channels = input.shape.back();
    output.resize_discard(input.shape);
    cudaRowSoftmax<Type><<<CudaGetBlocks(batch_size), BLOCK_SIZE>>>(input.p_data, output.p_data, batch_size, channels);
}

static ConvShape MakeConvShape(const TinyTensor<float>& input, const TinyTensor<float>& weight,
                               int pad_h, int pad_w, int stride_h, int stride_w){
    int batch_size = 1;
    for (int i = 0; i < (int)input.shape.size() - 3; i++) {
        batch_size *= input.shape[i];
    }
    int channels_in = input.shape[input.shape.size() - 3];
    int im_h = input.shape[input.shape.size() - 2];
    int im_w = input.shape.back();
    int kernel_h = weight.shape[weight.shape.size() - 2];
    int kernel_w = weight.shape.back();
    if (weight.shape[1] != channels_in || !UseDirectConv(channels_in * kernel_h * kernel_w)){
        throw std::invalid_argument("ConvReluMaxPool: unsupported shape (needs Cin * kh * kw <= 256)");
    }
    int col_h = (im_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int col_w = (im_w + 2 * pad_w - kernel_w) / stride_w + 1;
    return ConvShape{batch_size, channels_in, weight.shape[0], im_h, im_w, col_h, col_w, kernel_h, kernel_w,
                     pad_h, pad_w, stride_h, stride_w};
}

// conv + ReLU + 2x2 最大池化（步长 2）融合层：卷积结果不写回显存，
// pooled 为 [N, Cout, out_h / 2, out_w / 2]，argmax 记录每个池化窗口里最大值的位置（反向时使用）
void ConvReluMaxPoolForward(
    const TinyTensor<float>& input,
    const TinyTensor<float>& weight,
    TinyTensor<float>& pooled,
    TinyTensor<int>& argmax,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
){
    ConvShape s = MakeConvShape(input, weight, pad_h, pad_w, stride_h, stride_w);
    std::vector<int> shape{s.batch, s.c_out, s.out_h / 2, s.out_w / 2};
    pooled.resize_discard(shape);
    argmax.resize_discard(shape);
    ConvReluPoolForward(input.p_data, weight.p_data, pooled.p_data, argmax.p_data, s, shape[2], shape[3]);
}

// 反向：先由池化梯度还原卷积输出的梯度（一个 kernel，不需要清零和 atomicAdd），再算卷积核和输入的梯度
void ConvReluMaxPoolBackward(
    const TinyTensor<float>& input,
    const TinyTensor<float>& weight,
    const TinyTensor<float>& grad_pooled,
    const TinyTensor<int>& argmax,
    TinyTensor<float>& grad_input,
    TinyTensor<float>& grad_weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    bool need_grad_input
){
    ConvShape s = MakeConvShape(input, weight, pad_h, pad_w, stride_h, stride_w);
    TinyTensor<float> grad_conv{std::vector<int>{s.batch, s.c_out, s.out_h, s.out_w}, "gpu"};
    grad_weight.resize_discard(weight.shape);
    ReluPoolBackward(grad_pooled.p_data, argmax.p_data, grad_conv.p_data, s, s.out_h / 2, s.out_w / 2,
                     grad_weight.p_data, (int)grad_weight.size);
    ConvDirectBackwardWeight(input.p_data, grad_conv.p_data, grad_weight, s, true);
    if (need_grad_input){
        grad_input.resize_discard(input.shape);
        ConvDirectBackwardInput(grad_conv.p_data, weight.p_data, grad_input.p_data, s);
    }
}

// softmax 交叉熵（一个 kernel）：loss = mean_i -log softmax(z_i)[y_i]，grad = (softmax - onehot) / N
void SoftmaxCrossEntropy(
    const TinyTensor<float>& input,
    const TinyTensor<int>& labels,
    TinyTensor<float>& loss,
    TinyTensor<float>& grad
){
    int batch_size = 1;
    for (int i = 0; i < (int)input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
    }
    int channels = input.shape.back();
    loss.resize_discard(std::vector<int>{1});
    grad.resize_discard(input.shape);
    cudaSoftmaxCrossEntropy<float><<<1, BLOCK_SIZE>>>(input.p_data, labels.p_data, loss.p_data, grad.p_data,
                                                     batch_size, channels, 1.0f / batch_size);
}

// 返回分类错误率（不是 loss）
template <typename Type>
float SoftmaxLoss(
    const TinyTensor<Type>& softmax_output,
    const TinyTensor<int>& labels
){
    int batch_size = 1;
    for (int i = 0; i < (int)softmax_output.shape.size() - 1; i++) {
        batch_size *= softmax_output.shape[i];
    }
    int channels = softmax_output.shape.back();

    std::vector<Type> prob = softmax_output.get_data();
    std::vector<int> real = labels.get_data();
    float wrong = 0;
    for (int i = 0; i < batch_size; i++){
        int max_idx = 0;
        for (int j = 1; j < channels; j++){
            if (prob[i * channels + j] > prob[i * channels + max_idx]){
                max_idx = j;
            }
        }
        if (max_idx != real[i]){
            wrong += 1;
        }
    }
    return wrong / batch_size;
}

// loss 中存放整个 batch 的交叉熵之和
template <typename Type>
void CrossEntropyLoss(
    const TinyTensor<Type>& input,
    const TinyTensor<int>& labels,
    TinyTensor<Type>& loss
){
    int batch_size = 1;
    for (int i = 0; i < (int)input.shape.size() - 1; i++) {
        batch_size *= input.shape[i];
    }
    int channels = input.shape.back();

    TinyTensor<Type> temp{std::vector<int>{batch_size}, "gpu"};
    loss.resize_discard(std::vector<int>{1});
    cudaChannelLog<Type><<<CudaGetBlocks(batch_size), BLOCK_SIZE>>>(input.p_data, temp.p_data, labels.p_data, batch_size, channels);
    cudaSumToScalar<Type><<<1, BLOCK_SIZE>>>(temp.p_data, loss.p_data, batch_size, Type(1));
}

// 对 logits 的梯度：softmax - onehot（未除以 batch size）
template <typename Type>
void CrossEntropyLossBackward(
    const TinyTensor<Type>& softmax_output,
    const TinyTensor<int>& labels,
    TinyTensor<Type>& grad_input
){
    int batch_size = 1;
    for (int i = 0; i < (int)softmax_output.shape.size() - 1; i++) {
        batch_size *= softmax_output.shape[i];
    }
    int channels = softmax_output.shape.back();
    grad_input.resize_discard(softmax_output.shape);
    cudaSoftmaxGrad<Type><<<CudaGetBlocks(batch_size * channels), BLOCK_SIZE>>>(softmax_output.p_data, labels.p_data, grad_input.p_data, batch_size, channels);
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
    TinyTensor<float>& grad_bias,
    bool need_grad_input
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
    const int stride_w,
    bool need_grad_input
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