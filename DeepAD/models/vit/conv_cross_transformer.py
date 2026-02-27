import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np
from einops import rearrange


class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.weight = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            # nn.LeakyReLU(0.1, inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            # nn.Sigmoid()
        )

    def forward(self, x, y):
        # Ensure x and y are scaled
        concatenated = torch.cat([x, y], dim=1)
        weight = self.weight(concatenated)
        gate_values = self.gate(concatenated)
        x_gated = gate_values * weight * x
        y_gated = (1 - gate_values) * weight * y
        return x_gated, y_gated


class WeightedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(WeightedFusion, self).__init__()
        self.weight1 = nn.Linear(embed_dim, embed_dim)
        self.weight2 = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.embed_dim = embed_dim

    def forward(self, x, y):
        # Ensure x and y are scaled
        norm_x = self.ln1(x)
        norm_y = self.ln2(y)
        weight1 = self.leakyrelu(self.weight1(norm_x))
        weight2 = self.leakyrelu(self.weight2(norm_y))
        x_weighted = weight1 * norm_x + norm_x
        y_weighted = weight2 * norm_y + norm_y

        return x_weighted, y_weighted
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, resolution, split_size=7, dim_out=None, 
                 num_heads=8, focusing_factor=3, kernel_size=5):
        super(MultiHeadedAttention, self).__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.H = resolution
        self.W = resolution
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H * self.W, dim)))

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H * W).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H * W).permute(0, 2, 1).contiguous()
        return x, lepe
    
    def forward(self, qkv):

        q, k, v = qkv[0], qkv[1], qkv[2]
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        v, lepe = self.get_lepe(v, self.get_v)
        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H, w=self.W)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        # Gated fusion between original taskmap and attention output
        return x

class Fusion_Transformer(nn.Module):
    def __init__(self, dim, reso=224 // 4, num_heads=8, mlp_ratio=4.,
                 drop=0.1, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, focusing_factor=3, kernel_size=5):
        super(Fusion_Transformer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.patches_resolution = reso 
        self.mlp_ratio = mlp_ratio

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.proj = nn.Linear(dim, dim)

        self.cross_attn = MultiHeadedAttention(dim, resolution=self.patches_resolution, num_heads=num_heads, 
                                               dim_out=dim, focusing_factor=focusing_factor, 
                                               kernel_size=kernel_size)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, taskmap, heatmap):
        H = W = self.patches_resolution
        B, L, C = taskmap.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        taskmap = self.norm1(taskmap)
        heatmap = self.norm2(heatmap)
        q = self.q_proj(taskmap).reshape(B, -1, 1, C)
        k = self.k_proj(heatmap).reshape(B, -1, 1, C)
        v = self.v_proj(heatmap).reshape(B, -1, 1, C)
        qkv = torch.cat([q, k, v], dim=2).permute(2, 0, 1, 3)

        attn_out = self.cross_attn(qkv)
        attn_out = self.proj(attn_out)
        
        attn_out = attn_out + self.drop_path(attn_out)
        attn_out = attn_out + self.drop_path(self.mlp(self.norm3(attn_out)))

        return attn_out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# if __name__ == "__main__":
#     from torchsummary import summary
#     from torchvision.models import resnet50

#     # 假设输入数据的分辨率为224x224
#     input_resolution = 32 
#     dim = 32

#     # 创建Fusion_Transformer模型
#     model = Fusion_Transformer(dim=dim, reso=input_resolution).cuda()

#     # 打印模型参数量和计算量
#     summary(model, input_size=[(input_resolution**2, dim), (input_resolution**2, dim)])
