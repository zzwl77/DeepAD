import torch
import torch.nn as nn
from models.utils import *
# from models.vit.cross_transformer import Fusion_Transformer
from models.vit.conv_cross_transformer import Fusion_Transformer
from models.vit.flatten_cswin import Merge_Block
from einops.layers.torch import Rearrange


__all__ = ['ac2']

def ac2(**kwargs):
    """
        Only left.
    """
    model = NBModel(**kwargs)
    return model

        
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_features):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_features // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.merge = nn.Sequential(nn.Conv2d(out_features // 8, out_features // 8, 7, 4, 2),
                                   Rearrange('b c h w -> b (h w) c', h=224 // 4, w=224 // 4),
                                   nn.LayerNorm(out_features // 8),
                                   Merge_Block(dim=out_features // 8, dim_out=out_features // 4),
                                #    nn.LayerNorm(out_features // 4),
                                   Merge_Block(dim=out_features // 4, dim_out=out_features // 2),
                                #    nn.LayerNorm(out_features // 2),
                                   Merge_Block(dim=out_features // 2, dim_out=out_features),
                                #    nn.LayerNorm(out_features),
                                   )
        
        
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_features, out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.merge(out) ## [batch_size, 7 * 7, out_features]
        return out
    

class SingleTaskFusion(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_heads=8, dropout=0):
        super(SingleTaskFusion, self).__init__()
        self.feature1 = FeatureExtractor(in_channels, embedding_dim)
        self.feature2 = FeatureExtractor(in_channels, embedding_dim)
        self.stereo_fusion = 'average'  # 可选的融合方法
        self.multimodal_fusion = Fusion_Transformer(dim=embedding_dim, reso=7)
        self.weight = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_channels * 2, 2, kernel_size=1, stride=1, padding=0))
    
    def fuse_heatmaps(self, left_heatmap, right_heatmap, method='weighted'):
        if method == 'average':
            return (left_heatmap + right_heatmap) / 2
        elif method == 'max':
            return torch.max(left_heatmap, right_heatmap)
        elif method == 'weighted':
            weight = self.weight(torch.cat((left_heatmap, right_heatmap), dim=1))
            weighted_feature = weight[:, :1, :, :] * left_heatmap + weight[:, 1:, :, :] * right_heatmap
            
            # 实现加权平均融合策略
            return weighted_feature
        else:
            raise ValueError('Unsupported fusion method: {}'.format(method))

    def forward_single(self, taskmap, left_heatmap, right_heatmap):
        # Fuse left and right heatmaps
        heatmap = self.fuse_heatmaps(left_heatmap, left_heatmap, method=self.stereo_fusion)
        # Extract features
        task_features = self.feature1(taskmap)
        heat_features = self.feature2(heatmap)
        
        # Feature fusion between taskmap and heatmap
        combined_features = self.multimodal_fusion(task_features, heat_features)

        return combined_features
    
    def forward(self, taskmap, heatmap):
        # taskmap, heatmap shape: [batch_size, num_tasks, channels, height, width]
        combined_features = []
        for i in range(len(taskmap)):
            left_heatmap = heatmap[2*i]
            right_heatmap = heatmap[2*i + 1]
            task_feature = self.forward_single(taskmap[i], left_heatmap, right_heatmap)
            B, new_HW, C = task_feature.shape
            H = W = int(new_HW ** 0.5)
            task_feature = task_feature.transpose(-2, -1).contiguous().view(B, C, H, W)
            combined_features.append(task_feature)
            
        # 返回所有任务的特征    
        return torch.stack(combined_features, dim=1)
    

# class TemporalAttention(nn.Module):
#     def __init__(self, seq_len, c):
#         super(TemporalAttention, self).__init__()
#         self.conv = nn.Conv2d(seq_len * c, c, kernel_size=3, padding=1)
#         self.attention = SE(seq_len, c)
#         self.bn = nn.BatchNorm2d(c)

#     def forward(self, seq_features):
#         # seq_features shape: [batch_size, seq_len, c, h, w]
#         batch_size, seq_len, c, h, w = seq_features.shape
#         x = seq_features.view(batch_size, seq_len * c, h, w)  
#         x = self.conv(x)      # x shape: [batch_size, c, h, w]
#         weights = self.attention(x)
#         weighted_x = weights * x + x
#         weighted_x = self.bn(weighted_x)
#         return weighted_x

class SE(nn.Module):
    def __init__(self, seq_len, out_planes):
        super(SE, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        # Removed bias from convolutions for better performance
        self.fc = nn.Linear(out_planes, out_planes // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_planes // 4, out_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.GAP(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2).unsqueeze(3)
        return out
    

class TemporalAttention(nn.Module):
    def __init__(self, seq_len, c):
        super(TemporalAttention, self).__init__()
        self.conv = nn.Conv2d(seq_len * c, c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(seq_len * c, c, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1)
        self.attention = SE(seq_len, c)
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
    
    def forward(self, seq_features1, seq_features2):
        # seq_features shape: [batch_size, seq_len, c, h, w]
        batch_size, seq_len, c, h, w = seq_features1.shape
        x = seq_features2.view(batch_size, seq_len * c, h, w)
        x = self.conv(x)
        x = self.lrelu(x)
        x = self.bn1(x)
        weights = self.attention(x)

        x2 = seq_features1.view(batch_size, seq_len * c, h, w)
        x2 = self.conv2(x2)
        # x2 = self.lrelu(x2)
        x2 = self.bn2(x2)

        weighted_x = weights * x2 + x2
        
        
        return weighted_x
                                       
class crosstemporalFusion(nn.Module):
    """帧间融合模块"""
    def __init__(self, embed_dim, hidden_dim):
        super(crosstemporalFusion, self).__init__()
        self.fusion_layer = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, current_feature, previous_feature):
        ## current_feature, previous_feature shape: [batch_size, embed_dim, height, width]
        if previous_feature is None:
            previous_feature = torch.zeros_like(current_feature)
        
        # 将当前特征和前一特征沿着维度1（通道维度）拼接
        combined_feature = torch.cat([current_feature, previous_feature], dim=1)
        # 使用卷积层融合特征
        x = self.fusion_layer(combined_feature)
        return x
    

class BiGatedFusion(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim):
        super(BiGatedFusion, self).__init__()
        # 正向融合模块
        self.forward_fusion_module = nn.ModuleList([crosstemporalFusion(input_dim, hidden_dim) for _ in range(seq_len)])
        # 反向融合模块
        self.backward_fusion_module = nn.ModuleList([crosstemporalFusion(input_dim, hidden_dim) for _ in range(seq_len)])
        # self.feedback_attention = TemporalAttention(seq_len, input_dim, hidden_dim)
        self.attention = TemporalAttention(seq_len, input_dim)

    def forward(self, sequence):
        # sequence 形状：[batch_size, seq_len, input_dim]
        batch_size, seq_len, _, _ , _ = sequence.shape
        # 初始化正向和反向的输出列表
        forward_outputs = []
        backward_outputs = [None] * seq_len # 为反向输出预留空间
        
        # 正向迭代
        forward_prev = None
        for i in range(seq_len):
            current_feature = sequence[:, i, :, :, :]
            forward_output = self.forward_fusion_module[i](current_feature, forward_prev)
            forward_prev = forward_output
            forward_outputs.append(forward_output)
            # if i == seq_len - 1:
            #     forward_outputs = forward_output
        
        # 反向迭代
        backward_next = None
        for i in reversed(range(seq_len)):
            current_feature = sequence[:, i, :, :, :]
            backward_output = self.backward_fusion_module[i](current_feature, backward_next)
            backward_next = backward_output
            backward_outputs[i] = backward_output
            # if i == 0:
            #     backward_outputs = backward_output

        combined_outputs = [(f + b) / 2 for f, b in zip(forward_outputs, backward_outputs)]
        combined_outputs = torch.stack(combined_outputs, dim=1)
        combined_outputs = self.attention(sequence, combined_outputs)
        # 将列表转换为张量
        return combined_outputs
    

class TemporalFusion(nn.Module):
    """时序融合网络，每个局部时序特征融合有独立的参数"""
    def __init__(self, embed_dim, hidden_dim):
        super(TemporalFusion, self).__init__()
        self.short_term_fusion_modules = nn.ModuleList([BiGatedFusion(4, embed_dim, hidden_dim),
                                                        BiGatedFusion(3, embed_dim, hidden_dim),
                                                        BiGatedFusion(5, embed_dim, hidden_dim)])
        self.long_term_fusion = BiGatedFusion(3, hidden_dim, hidden_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MlpHead(hidden_dim, 1)

    def forward(self, x_list):
        # x_list包含四个时序特征组，每个形状为(batch_size, num_tasks_in_group, in_channels)
        short_term_outputs = []

        # 对每个时序特征组进行独立的短期时序融合
        for idx, x in enumerate(x_list):
            fused_feature = self.short_term_fusion_modules[idx](x)
            short_term_outputs.append(fused_feature)  # 移除单一时间维度
        # 准备长期时序融合的输入
        long_term_input = torch.stack(short_term_outputs, dim=1)  # 维度调整以匹配长期融合模块
        # 长期时序融合
        fused_feature = self.long_term_fusion(long_term_input)
        # 全局平均池化
        fused_feature = self.avgpool(fused_feature)
        # 展平并应用全连接层
        fused_feature = fused_feature.view(fused_feature.size(0), -1)
        fused_feature = self.fc(fused_feature).squeeze(-1)
        
        return fused_feature # 移除单一维度
    

# class ParamGenerator(nn.Module):
#     def __init__(self):
#         super(ParamGenerator, self).__init__()
#         self.fc = nn.Linear(2, 2)  # 输出两个值：scale 和 shift
#         self.bn = nn.BatchNorm1d(2)
#         self.relu = nn.ReLU()

#     def forward(self, age_education):
#         x = self.fc(age_education)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x  # 返回 scale 和 shift
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    

class MlpHead(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(MlpHead, self).__init__()
        # Removed bias from convolutions for better performance
        self.fc1 = nn.Linear(in_planes, in_planes // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_planes // 2, out_planes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class NBModel(nn.Module):
    def __init__(self, in_channels=3, task_embedding_dim=64, hidden_dim=64):
        super(NBModel, self).__init__()
        self.task_fusion1 = SingleTaskFusion(in_channels, task_embedding_dim, dropout=0.1)
        self.task_fusion2 = SingleTaskFusion(in_channels, task_embedding_dim, dropout=0.1)
        self.task_fusion3 = SingleTaskFusion(in_channels, task_embedding_dim, dropout=0.1)
        self.sequence_fusion = TemporalFusion(task_embedding_dim, hidden_dim)
        # self.sequence_fusion = GlobalFeatureExtractor(12 * task_embedding_dim)
        # self.param_generator = ParamGenerator()  # 假设融合后特征的维度为 hidden_dim
        
        self.weight_init(self)

    def forward(self, heatmaps, taskmaps, age_education):
        # Process each task
        
        task_feature1 = self.task_fusion1(taskmaps[:4], heatmaps[:8])
        task_feature2 = self.task_fusion2(taskmaps[4:7], heatmaps[8:14])
        task_feature3 = self.task_fusion3(taskmaps[7:12], heatmaps[14:24])
        # print(task_feature1.shape, task_feature2.shape, task_feature3.shape)

        out = self.sequence_fusion([task_feature1, task_feature2, task_feature3])  # [batch_size, 1]
        # out = self.adapt_pool(out)  # Global average pooling
        # out = torch.flatten(out, 1)
        # # print(out.shape)
        # out = self.projecter(out).squeeze(-1)
        # # scale, shift = self.param_generator(age_education).chunk(2, dim=-1)
        # # score = fused_feature * scale + shift + fused_feature
        # # score = self.projecter(fused_feature).squeeze(-1)
        return out  # Remove the last dimension for consistency

    @staticmethod
    def weight_init(net, init_typer='kaiming', init_gain = 0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_typer == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_typer == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_typer == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_typer == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_typer)
            elif classname.find('BatchNorm2d') != -1 or classname.find('LayerNorm') != -1:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)  
        net.apply(init_func)

# from torchinfo import summary
# from thop import clever_format, profile

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 假设输入数据的维度是 [channels, height, width]
#     in_channels = 3  # 示例输入通道数
#     height, width = 224, 224  # 示例输入尺寸
#     task_embedding_dim = 32  # 示例任务嵌入维度
#     hidden_dim = 32  # 示例隐藏层维度

#     # 初始化模型并移动到适当的设备
#     model = NBModel(in_channels, task_embedding_dim, hidden_dim).to(device)

#     # 假定输入数据的批量大小为1
#     taskmaps = [torch.randn(1, in_channels, height, width).to(device) for _ in range(16)]  # 16个任务图
#     heatmaps = [torch.randn(1, in_channels, height, width).to(device) for _ in range(32)]  # 32个热图

#     # 使用torchsummary输出模型概要信息
#     # summary(model, input_size=[(in_channels, height, width)] * 48)  # 16个任务图 + 32个热图

#     # 使用thop计算FLOPs
#     input = [taskmaps, heatmaps]
#     flops, params = profile(model, inputs=(taskmaps, heatmaps), verbose=False)
#     flops, params = clever_format([flops, params], "%.3f")

#     print(f"Model FLOPs: {flops}")
#     print(f"Model Parameters: {params}")

# if __name__ == "__main__":
#     main()