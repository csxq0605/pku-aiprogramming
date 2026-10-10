// myNN：图像模型用的算子（主机端接口）。T = float / bf16，权重梯度、归一化参数与统计量一律 fp32。
// 所有输出张量由函数内部 resize_discard 成正确形状；带 accumulate 的梯度直接累加到（通常是一整块扁平梯度
// 缓冲区里的）视图上，每个 step 开头统一清零一次。
#ifndef NN_OPS_H
#define NN_OPS_H

#include <vector>
#include "nn_common.cuh"

// ---------------- 卷积（NHWC，隐式 GEMM）----------------
template <typename T>
void ConvForward(const TinyTensor<T>& x, const TinyTensor<T>& w, TinyTensor<T>& y,
                 int pad_h, int pad_w, int stride_h, int stride_w, int groups);
template <typename T>
void ConvForwardImpl(const T* x, const T* w, T* y, const ConvGeom& g);
// bf16 tensor core 隐式 GEMM（conv_tc.cu），mode: 0 前向 / 1 输入梯度 / 2 权重梯度（累加到 fp32）
bool ConvTensorCoreSupported(const ConvGeom& g, int mode);
void ConvTensorCore(int mode, const bf16* a, const bf16* b, void* out, const ConvGeom& g);
// fp32 大 tile 隐式 GEMM（conv_f32.cu）
bool ConvF32Supported(const ConvGeom& g, int mode);
void ConvF32(int mode, const float* a, const float* b, float* out, const ConvGeom& g);
// 每组 4 / 8 / 16 个通道的分组卷积（conv_grouped.cu），参数含义同上
bool ConvGroupedSupported(const ConvGeom& g);
template <typename T>
void ConvGrouped(int mode, const T* a, const T* b, void* out, const ConvGeom& g);
// 3 通道输入的 3x3 步长 1 卷积（第一层，conv_stem.cu）：mode 0 前向 (a=x, b=w) / 2 权重梯度 (a=dy, b=x)
bool ConvStemSupported(const ConvGeom& g);
template <typename T>
void ConvStem(int mode, const T* a, const T* b, void* out, const ConvGeom& g);

template <typename T>
void ConvBackwardData(const TinyTensor<T>& dy, const TinyTensor<T>& w, TinyTensor<T>& dx, const std::vector<int>& x_shape,
                      int pad_h, int pad_w, int stride_h, int stride_w, int groups);
template <typename T>
void ConvBackwardWeight(const TinyTensor<T>& x, const TinyTensor<T>& dy, TinyTensor<float>& dw,
                        int pad_h, int pad_w, int stride_h, int stride_w, int groups, bool accumulate);

// ---------------- BatchNorm（按最后一维 = 通道归一化），可选融合残差相加与 ReLU ----------------
// 训练：y = act(gamma * (x - mean_B) / sqrt(var_B + eps) + beta [+ residual])
// stats 保存 [sum(C), sumsq(C)]，反向直接用它重算均值方差；同时更新 running_mean / running_var（无偏方差）
template <typename T>
void BatchNormTrain(const TinyTensor<T>& x, const TinyTensor<float>& gamma, const TinyTensor<float>& beta,
                    TinyTensor<float>& running_mean, TinyTensor<float>& running_var, TinyTensor<float>& stats,
                    TinyTensor<T>& y, const TinyTensor<T>* residual, bool relu, float momentum, float eps);
template <typename T>
void BatchNormEval(const TinyTensor<T>& x, const TinyTensor<float>& gamma, const TinyTensor<float>& beta,
                   const TinyTensor<float>& running_mean, const TinyTensor<float>& running_var,
                   TinyTensor<T>& y, const TinyTensor<T>* residual, bool relu, float eps);
// 反向：g = dy * [y > 0]（relu 时）；dresidual = g（需要时）；dgamma / dbeta 累加
template <typename T>
void BatchNormBackward(const TinyTensor<T>& x, const TinyTensor<T>& y, const TinyTensor<T>& dy,
                       const TinyTensor<float>& gamma, const TinyTensor<float>& stats,
                       TinyTensor<T>& dx, TinyTensor<float>& dgamma, TinyTensor<float>& dbeta,
                       TinyTensor<T>* dresidual, bool relu, float eps);

// ---------------- LayerNorm（最后一维）----------------
template <typename T>
void LayerNormForward(const TinyTensor<T>& x, const TinyTensor<float>& gamma, const TinyTensor<float>& beta,
                      TinyTensor<T>& y, TinyTensor<float>& mean_rstd, float eps);
template <typename T>
void LayerNormBackward(const TinyTensor<T>& x, const TinyTensor<T>& dy, const TinyTensor<float>& gamma,
                       const TinyTensor<float>& mean_rstd, TinyTensor<T>& dx,
                       TinyTensor<float>& dgamma, TinyTensor<float>& dbeta);

// ---------------- 池化 ----------------
template <typename T>
void MaxPool2x2Forward(const TinyTensor<T>& x, TinyTensor<T>& y, TinyTensor<unsigned char>& argmax);
template <typename T>
void MaxPool2x2Backward(const TinyTensor<T>& dy, const TinyTensor<unsigned char>& argmax, TinyTensor<T>& dx,
                        const std::vector<int>& x_shape);
template <typename T>
void GlobalAvgPoolForward(const TinyTensor<T>& x, TinyTensor<T>& y);
template <typename T>
void GlobalAvgPoolBackward(const TinyTensor<T>& dy, TinyTensor<T>& dx, const std::vector<int>& x_shape);

// ---------------- 全连接：y = act(x W^T + b)，W [out, in]；act 0=无 1=ReLU 2=GELU(erf) ----------------
// GELU 时 pre 保存激活前的值供反向使用。OutT 可为 float（混合精度下的最后一层 logits）
template <typename T, typename OutT>
void LinearForward(const TinyTensor<T>& x, const TinyTensor<T>& w, const TinyTensor<float>* b, TinyTensor<OutT>& y,
                   int act, TinyTensor<T>* pre);
// dy 为 OutT；dx 不需要时传 nullptr；dw / db 累加（fp32）
template <typename T, typename OutT>
void LinearBackward(const TinyTensor<T>& x, const TinyTensor<T>& w, const TinyTensor<OutT>& y, const TinyTensor<OutT>& dy,
                    int act, const TinyTensor<T>* pre, TinyTensor<T>* dx, TinyTensor<float>& dw, TinyTensor<float>* db);

// ---------------- 注意力：qkv [B, L, 3, H, hd] -> out [B, L, H, hd]；probs [B, H, L, L] fp32 ----------------
// bf16、L <= 80、head 维 64 时的 tensor core 版本（attention_tc.cu）；probs 里只存每行的 logsumexp [B, H, L]
bool AttentionTcSupported(int L, int hd);
void AttentionTcForward(const bf16* qkv, bf16* out, float* lse, int B, int L, int H);
void AttentionTcBackward(const bf16* qkv, const float* lse, const bf16* dout, bf16* dqkv, int B, int L, int H);
template <typename T>
void AttentionForward(const TinyTensor<T>& qkv, int heads, TinyTensor<T>& out, TinyTensor<float>& probs);
template <typename T>
void AttentionBackward(const TinyTensor<T>& qkv, const TinyTensor<float>& probs, const TinyTensor<T>& dout, int heads,
                       TinyTensor<T>& dqkv);

// ---------------- ViT 的 token 处理 ----------------
// patches [B, N, D] -> out [B, N+1, D]：第 0 个 token 为 cls，再整体加位置编码 pos [N+1, D]
template <typename T>
void TokensForward(const TinyTensor<T>& patches, const TinyTensor<T>& cls, const TinyTensor<T>& pos, TinyTensor<T>& out);
template <typename T>
void TokensBackward(const TinyTensor<T>& dout, TinyTensor<T>& dpatches, TinyTensor<float>& dcls, TinyTensor<float>& dpos);
// x [B, L, D] -> y [B, D]（取第 0 个 token）及其反向
template <typename T>
void SelectTokenForward(const TinyTensor<T>& x, TinyTensor<T>& y);
template <typename T>
void SelectTokenBackward(const TinyTensor<T>& dy, TinyTensor<T>& dx, const std::vector<int>& x_shape);

// ---------------- 逐元素 ----------------
template <typename T>
void AddForward(const TinyTensor<T>& a, const TinyTensor<T>& b, TinyTensor<T>& out);
// 单独的激活 y = act(x)，act 1 relu / 2 gelu；反向 dx = dy * act'(x)（不融合的消融实验用）
template <typename T>
void ActForward(const TinyTensor<T>& x, TinyTensor<T>& y, int act);
template <typename T>
void ActBackward(const TinyTensor<T>& x, const TinyTensor<T>& dy, TinyTensor<T>& dx, int act);
template <typename Src, typename Dst>
void CastTensor(const TinyTensor<Src>& src, TinyTensor<Dst>& dst);

// ---------------- 损失：logits fp32 [N, C]；loss_sum[0] += sum(-log p_y)，correct[0] += 预测正确数 ----------------
// grad = (softmax - onehot) * grad_scale（通常为 1/N）；loss_sum / correct 由调用方清零（跨多个 batch 累加）
void SoftmaxCrossEntropyForwardBackward(const TinyTensor<float>& logits, const TinyTensor<int>& labels,
                                        TinyTensor<float>& grad, float grad_scale,
                                        TinyTensor<float>& loss_sum, TinyTensor<int>& correct, bool want_grad);

// ---------------- 数据：常驻 GPU 的 uint8 图像，按 GPU 上的步数取 batch 并做数据增强 ----------------
// step: [当前 epoch 内的 batch 序号, 内部计数, 全局步数, 0]；kernel 结束后 [0] 和 [2] 各加 1
// augment 时：四周补 pad 个 0 像素后随机裁回原尺寸 + 随机水平翻转，随机数 = hash(seed, 全局步数, 样本序号)
template <typename T>
void LoadBatch(const TinyTensor<unsigned char>& images, const TinyTensor<int>& labels, const TinyTensor<int>& order,
               TinyTensor<int>& step, TinyTensor<T>& x, TinyTensor<int>& y, int batch,
               const std::vector<float>& mean, const std::vector<float>& std, bool augment, int pad,
               unsigned int seed);

// ---------------- 优化器（作用在扁平的参数 / 梯度缓冲区上，一个 kernel）----------------
// 学习率按 GPU 上的全局步数 t = step[2] - 1 计算：前 warmup 步线性升温，之后余弦退火到 base_lr * final_ratio
struct LrSchedule{
    float base_lr;
    int warmup_steps;
    int total_steps;
    float final_ratio;
};
// w_lowp 非空时同时写出 bf16 权重副本（混合精度前向用）
void SgdMomentumStep(TinyTensor<float>& w, const TinyTensor<float>& g, TinyTensor<float>& m, TinyTensor<bf16>* w_lowp,
                     const TinyTensor<int>& step, LrSchedule lr, float momentum, float weight_decay, bool nesterov);
void AdamWStep(TinyTensor<float>& w, const TinyTensor<float>& g, TinyTensor<float>& m, TinyTensor<float>& v,
               TinyTensor<bf16>* w_lowp, const TinyTensor<int>& step, LrSchedule lr, float beta1, float beta2,
               float eps, float weight_decay);

#endif
