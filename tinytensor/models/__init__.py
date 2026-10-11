"""
模型：MNIST 上的 MLP / CNN（contiguous_format），64x64 图像上的 VGG16 / ResNet18 / ResNeXt26 / ViT-Tiny（channels_last）
"""
from .mnist import CNN, MLP  # noqa: F401
from .resnet import ResNet18, ResNeXt26  # noqa: F401
from .vgg import VGG16, ImageModel  # noqa: F401
from .vit import ViTTiny  # noqa: F401

MODELS = {m.name: m for m in (VGG16, ResNet18, ResNeXt26, ViTTiny)}
