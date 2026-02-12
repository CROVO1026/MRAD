import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from collections import OrderedDict
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class AnomalyMLP(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.mlp1 = nn.Linear(d_model, d_model)
        self.mlp2 = nn.Linear(d_model, d_model)
    def forward(self, patch_emb1,patch_emb2): 
        # 预测目标角度
        return torch.stack([self.mlp1(patch_emb1), self.mlp2(patch_emb2)], dim=1)
# class AnomalyMLP(nn.Module):
#     def __init__(self, d_model=768):
#         super().__init__()
#         # 流形学习网络
#         self.mlp1 = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.mlp2 = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),  
#             ("c_proj", nn.Linear(d_model * 4, d_model))]))

#     def forward(self, patch_emb): 
#         # 预测目标角度
#         return torch.stack([self.mlp1(patch_emb), self.mlp2(patch_emb)], dim=1)
class MLP(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.mlp1 = nn.Linear(d_model, d_model)
        self.mlp2 = nn.Linear(d_model, d_model)
        # self.mlp1 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.mlp2 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),  
        #     ("c_proj", nn.Linear(d_model * 4, d_model))]))
    def forward(self, emb1, emb2):
        return self.mlp1(emb1), self.mlp2(emb2)
class Projector(nn.Module):
    def __init__(self, dim_in, dim_out, length):
        super(Projector, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(length)])
        # self.apply(weights_init)

    def forward(self, emb,idx):
        return self.fc[idx](emb)
class Projector_multi_layer(nn.Module):
    def __init__(self, dim_in, dim_out, length, depth=1, activation=nn.ReLU):
        super(Projector_multi_layer, self).__init__()
        self.depth = max(1, depth)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation

        self.fc = nn.ModuleList([self._build_branch() for _ in range(length)])

    def _build_branch(self):
        layers = []
        in_dim = self.dim_in
        for layer_idx in range(self.depth):
            out_dim = self.dim_out
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx != self.depth - 1:
                layers.append(self.activation())
            in_dim = out_dim
        return nn.Sequential(*layers)

    def forward(self, emb, idx):
        return self.fc[idx](emb)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def avg_pool2d_reflect(x, k):
    """
    x : (B, C, H, W)
    k : kernel_size，必须为奇数才能“same”输出
    返回与 x 同分辨率的镜像平均池化
    """
    pad = k // 2  # 与零填充写法一致
    # F.pad 接收顺序是 (left, right, top, bottom)
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)

def avg_pool2d_reflect(x, k):
    """
    x : (B, C, H, W)
    k : kernel_size，必须为奇数才能“same”输出
    返回与 x 同分辨率的镜像平均池化
    """
    pad = k // 2  # 与零填充写法一致
    # F.pad 接收顺序是 (left, right, top, bottom)
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)

def average_neighbor(x, K=(1, 3, 5)):
    """
    对每个 token 做 1×1、3×3、5×5 平均池化并叠加（取均值）
    输入  : (B, N, C)  其中 N = H*W，H=W=sqrt(N)
    输出  : 同形状 (B, N, C)
    """
    B, N, C = x.shape
    H = W = int(N ** 0.5)
    x = x.transpose(1, 2).reshape(B, C, H, W)  # -> (B, C, H, W)

    outs = []
    for k in K:  # 多尺度窗口
        # outs.append(avg_pool2d_reflect(x, k))
        outs.append(F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2))
    out = sum(outs) / len(outs)  # 元素级平均叠加
    out = out.flatten(2).transpose(1, 2)  # -> (B, N, C)
    return out