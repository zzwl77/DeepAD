import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class CSWinBlock(nn.Module):
    def __init__(self, dim, reso=224 // 4, num_heads=2,
                 split_size=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0.1, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso                              
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.attns = nn.ModuleList([
                                        FocusedLinearAttention(
                                            dim // 2, resolution=self.patches_resolution, idx=i,
                                            split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                                            qk_scale=qk_scale, focusing_factor=focusing_factor, kernel_size=kernel_size)
                                        for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class CSWinBlock2(nn.Module):
    def __init__(self, dim, reso=224 // 4, num_heads=2,
                 split_size=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0.1, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso                              
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.attns = nn.ModuleList([
                                        FocusedLinearAttention(
                                            dim // 2, resolution=self.patches_resolution, idx=i,
                                            split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                                            qk_scale=qk_scale, focusing_factor=focusing_factor, kernel_size=kernel_size)
                                        for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x, y):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = y.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img1 = self.norm1(y)
        img2 = self.norm2(x)
        q = self.q(img1)
        kv = self.kv(img2)
        qkv = torch.cat([q, kv], dim=-1).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x
    
class Para_CSWinBlock(nn.Module):
    def __init__(self, dim, reso=224 // 4, num_heads=2,
                 split_size=1, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso                              
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.attns = nn.ModuleList([
                                        FocusedLinearAttention(
                                            dim // 2, resolution=self.patches_resolution, idx=i,
                                            split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                                            qk_scale=qk_scale, focusing_factor=focusing_factor, kernel_size=kernel_size)
                                        for i in range(self.branch_num)])


        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        mlped_x = self.mlp(img)
        x = attened_x + mlped_x + x

        return x
    
class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x
       
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
    
class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qk_scale=None, focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H_sp * self.W_sp, dim)))
        # print('Linear Attention {}x{} f{} kernel{}'.
        #       format(H_sp, W_sp, focusing_factor, kernel_size))

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        # x = x.reshape(-1, self.H_sp * self.W_sp, C).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, self.H_sp * self.W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)
        # q, k, v = (rearrange(x, "b h n c -> b n (h c)", h=self.num_heads) for x in [q, k, v])

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

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H_sp, w=self.W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)

        return x
    
def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img