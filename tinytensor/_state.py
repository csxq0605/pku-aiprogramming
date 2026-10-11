"""全局状态与常量（放在单独的模块里，避免子模块之间循环 import）"""
float32 = "fp32"
bfloat16 = "bf16"                       # 混合精度：激活值与卷积 / 全连接权重用 bf16，主权重和归一化层保持 fp32
contiguous_format = "contiguous_format"  # NCHW
channels_last = "channels_last"          # NHWC
seed = 0
grad_enabled = True
