#ifndef LAYERS_CUH
#define LAYERS_CUH

#include <cublas_v2.h>
#include "TinyTensor.h"
#include "base.h"

template <typename T>
void FcForward(
    const TinyTensor<T>& input,
    TinyTensor<T>& output,
    const TinyTensor<T>& weight,
    const TinyTensor<T>& bias
);

template <typename T>
void FcBackward(
    const TinyTensor<T>& input,
    const TinyTensor<T>& output,
    const TinyTensor<T>& weight,
    const TinyTensor<T>& bias,
    TinyTensor<T>& grad_input,
    const TinyTensor<T>& grad_output,
    TinyTensor<T>& grad_weight,
    TinyTensor<T>& grad_bais
);

template <typename T>
void im2col(
    const TinyTensor<T>& im_tensor,
    TinyTensor<T>& col_tensor,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename T>
void col2im(
    const TinyTensor<T>& col_tensor,
    TinyTensor<T>& im_tensor,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& im_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename T>
void ConvForward(
    const TinyTensor<T>& input,
    TinyTensor<T>& output,
    const TinyTensor<T>& weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename T>
void ConvBackward(
    const TinyTensor<T>& input,
    const TinyTensor<T>& output,
    const TinyTensor<T>& weight,
    TinyTensor<T>& grad_input,
    const TinyTensor<T>& grad_output,
    TinyTensor<T>& grad_weight,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename T>
void MaxPoolingForward(
    const TinyTensor<T>& input,
    TinyTensor<T>& output,
    TinyTensor<T>& mask,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename T>
void MaxPoolingBackward(
    const TinyTensor<T>& grad_output,
    const TinyTensor<T>& mask,
    TinyTensor<T>& grad_input,
    const std::vector<int>& input_shape,
    const std::vector<int>& kernel_shape,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w
);

template <typename T>
void SoftmaxForward(
    const TinyTensor<T>& input,
    TinyTensor<T>& output
);

template <typename T>
float SoftmaxLoss(
    const TinyTensor<T>& softmax_output,
    const TinyTensor<int>& labels
);

template <typename T>
void CrossEntropyLoss(
    const TinyTensor<T>& input,
    const TinyTensor<int>& labels,
    TinyTensor<T>& loss
);

template <typename T>
void CrossEntropyLossBackward(
    const TinyTensor<T>& softmax_output,
    const TinyTensor<int>& labels,
    TinyTensor<T>& grad_input
);

//#include "Layers.inl"
#endif