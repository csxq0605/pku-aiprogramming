import numpy as np

from .. import _state, nn
from ..nn import functional as F
from ..nn import intrinsic as nni
from .vgg import ImageModel


class Block(nn.Module):
    """pre-norm 的 Transformer block：x += proj(attention(qkv(ln1(x))))；x += fc2(gelu(fc1(ln2(x))))"""

    def __init__(self, dim, heads, mlp_ratio, fuse):
        super().__init__()
        self.dim, self.heads, self.fuse = dim, heads, fuse
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = (nni.LinearGELU if fuse else nn.Linear)(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        for layer in (self.qkv, self.proj, self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        B, L, D = x.shape
        H = self.heads
        y = self.qkv(self.ln1(x)).view(B, L, 3, H, D // H)
        y = self.proj(F.self_attention(y, H).view(B, L, D))
        x = x + y
        y = self.fc1(self.ln2(x))
        if not self.fuse:
            y = F.gelu(y)
        return x + self.fc2(y)


class ViTTiny(ImageModel):
    """ViT-Tiny：8x8 patch -> 64 个 token + cls，D=192，3 头，12 层 pre-norm block，MLP 比例 4"""
    name = "vit_tiny"
    image_size, patch_size, dim, heads, depth, mlp_ratio = 64, 8, 192, 3, 12, 4

    def build(self):
        D, P = self.dim, self.patch_size
        self.n_patches = (self.image_size // P) ** 2
        self.patch = nn.Conv2d(3, D, P, stride=P)
        nn.init.uniform_(self.patch.weight, np.sqrt(6.0 / (P * P * 3 + D)))
        self.cls = nn.init.std_uniform_(nn.Parameter((D,)))
        self.pos = nn.init.std_uniform_(nn.Parameter((self.n_patches + 1, D)))
        self.blocks = nn.ModuleList([Block(D, self.heads, self.mlp_ratio, self.fuse) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, self.num_classes, out_dtype=_state.float32)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = self.patch(x)                                              # [B, 8, 8, D]
        x = F.embed_tokens(x.view(x.shape[0], self.n_patches, self.dim), self.cls, self.pos)
        for block in self.blocks:
            x = block(x)
        return self.head(F.select_token(self.norm(x)))
