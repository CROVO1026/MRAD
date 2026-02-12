import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from collections import OrderedDict
class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key_value):
        # 交叉注意力：query作为查询，key_value作为键值对
        attn_output, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False
        )
        
        # 残差连接和层归一化
        query = self.norm1(query + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        
        return query
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionPooling(nn.Module):
    def __init__(self, dim=768, heads=1,query_num=2):
        super().__init__()
        self.query_num = query_num
        self.query = nn.Parameter(torch.randn(1, query_num, dim))  # Learnable query, shape: [1, 1, D]

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, patch_feats):
        """
        patch_feats: [B, N, D] where N is number of patches (e.g., 1369), D is feature dim (e.g., 768)
        Returns: [B, D]
        """
        B, N, D = patch_feats.shape
        H = self.heads
        Q = self.q_proj(self.query).expand(B,self.query_num , D)   # [B, 1, D]
        K = self.k_proj(patch_feats)                  # [B, N, D]
        V = self.v_proj(patch_feats)                  # [B, N, D]

        # Reshape for multi-head attention
        Q = Q.view(B, self.query_num, H, D // H).transpose(1, 2)    # [B, H, 1, D/H]
        K = K.view(B, N, H, D // H).transpose(1, 2)    # [B, H, N, D/H]
        V = V.view(B, N, H, D // H).transpose(1, 2)    # [B, H, N, D/H]


        attn = (Q @ K.transpose(-2, -1)) * self.scale # [B, H, 1, N]
        attn = attn.softmax(dim=-1)

        out = attn @ V                                # [B, H, 1, D/H]
        out = out.transpose(1, 2).reshape(B, self.query_num, D)     # [B, 1, D]
        out = self.out_proj(out)           # [B,query_num, D]

        return out

class NormalFeatureAttention(nn.Module):
    def __init__(self, dim=768, num_layers=4, num_heads=1, dropout=0.1, target_size=(518, 518),d_model=768):
        super().__init__()
        # 创建多层交叉注意力模块
        # self.cross_attention_layers = nn.ModuleList([
        #     CrossAttentionLayer(
        #         dim=dim,
        #         num_heads=num_heads,
        #         dropout=dropout
        #     ) for _ in range(num_layers)
        # ])
        # self.mlp1 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.mlp2 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        self.mlp1 = nn.Linear(d_model, d_model)
        self.mlp2 = nn.Linear(d_model, d_model)

        self.layer_norm1 = nn.LayerNorm(dim)
        self.target_size = target_size  # 插值目标分辨率

    def forward(self, patch_features, normal_feature, anomaly_feature):
        # patch_features: [batch_size, 1370, 768]
        # normal_feature: [1, 768]
        # anomaly_feature: [1, 768]
        
        batch_size = patch_features.size(0)
        
        # 步骤1：移除[CLS]标记
        patch_features = patch_features[:,0:, :]  # [：, :]  # [batch_size, 1369, 768]
        
        # 步骤2：创建独立的查询向量
        # 正常特征作为第一个查询
        # normal_query = normal_feature.expand(batch_size, 1, -1)  # [batch_size, 1, 768]
        # # 异常特征作为第二个查询
        # anomaly_query = anomaly_feature.expand(batch_size, 1, -1)  # [batch_size, 1, 768]
        normal_query = normal_feature
        anomaly_query = anomaly_feature  # [batch_size, 1, 768]
        nomal_sim = torch.matmul(normal_query,patch_features.transpose(1,2)) #[batch_size, 1, 1369]
        anomaly_sim = torch.matmul(anomaly_query,patch_features.transpose(1,2)) #[batch_size, 1, 1369]

        normal_attn_weights = F.softmax(nomal_sim, dim=-1) #[batch_size, 1, 1369]
        anomaly_attn_weights = F.softmax(anomaly_sim, dim=-1) #[batch_size, 1, 1369]

        normal_attn_output = torch.matmul(normal_attn_weights, patch_features) #[batch_size, 1, 768]
        anomaly_attn_output = torch.matmul(anomaly_attn_weights, patch_features) #[batch_size, 1, 768]

        normal_attn_output =normal_attn_output + normal_query  # 残差连接
        anomaly_attn_output = anomaly_attn_output + anomaly_query

        normal_attn_output = self.mlp1(normal_attn_output)
        anomaly_attn_output = self.mlp2(anomaly_attn_output)
        # # 拼接两个输出
        output = torch.cat([normal_attn_output, anomaly_attn_output], dim=1)  # [batch_size, 2, 768]
        output = self.layer_norm1(output)  # 层归一化
        return output # [batch_size, 2, 768]
        # # 分别处理正常和异常查询
        # normal_output = normal_query
        # anomaly_output = anomaly_query
        
        # # 步骤3：应用多层交叉注意力机制
        # # 只使用交叉注意力，没有自注意力
        # for layer in self.cross_attention_layers:
        #     # 正常特征与图像patch交互
        #     normal_output = layer(
        #         query=normal_output,
        #         key_value=patch_features_no_cls
        #     )
            
        #     # 异常特征与图像patch交互
        #     anomaly_output = layer(
        #         query=anomaly_output,
        #         key_value=patch_features_no_cls
        #     )
        
        # # 拼接两个输出
        # output = torch.cat([normal_output, anomaly_output], dim=1)  # [batch_size, 2, 768]
        # output = self.layer_norm(output)  # 层归一化
        



    


# class NormalFeatureAttention(nn.Module):
#     def __init__(self, dim=768, num_layers=2, num_heads=8, dropout=0.1, target_size=(518, 518)):
#         super().__init__()
#         # 创建Transformer解码器层用于交叉注意力
#         decoder_layer = TransformerDecoderLayer(
#             d_model=dim,
#             nhead=num_heads,
#             dim_feedforward=dim * 4,
#             dropout=dropout,
#             batch_first=True  # 使用batch_first格式
#         )
#         self.transformer_decoder = TransformerDecoder(
#             decoder_layer=decoder_layer,
#             num_layers=num_layers
#         )
#         self.layer_norm = nn.LayerNorm(dim)
#         self.target_size = target_size  # 插值目标分辨率

#     def forward(self, patch_features, normal_feature, anomaly_feature):
#         # patch_features: [batch_size, 1370, 768]
#         # normal_feature: [1, 768]
#         # anomaly_feature: [1, 768]
        
#         batch_size = patch_features.size(0)
        
#         # 步骤1：移除[CLS]标记
#         patch_features_no_cls = patch_features[:, 1:, :]  # [batch_size, 1369, 768]
        
#         # 步骤2：拼接normal和anomaly特征
#         target_features = torch.cat([
#             normal_feature, 
#             anomaly_feature
#         ], dim=0)  # [2, 768]
        
#         # 添加batch维度并扩展
#         target_features = target_features.unsqueeze(0)  # [1, 2, 768]
#         target_features = target_features.expand(batch_size, -1, -1)  # [batch_size, 2, 768]
        
#         # 步骤3：应用交叉注意力机制
#         # target_features作为query, patch_features_no_cls作为key/value
#         output = self.transformer_decoder(
#             tgt=target_features,
#             memory=patch_features_no_cls
#         )  # [batch_size, 2, 768]
        
#         output = self.layer_norm(output)  # 层归一化
        
        
#         return output # [batch_size, 2, 768]

#     def normalize_channel(self, tensor, channel_idx=1):
#         # 保持不变，但现在处理的是2通道数据
#         batch_size = tensor.size(0)
#         channel_data = tensor[:, channel_idx, :, :]  # [batch_size, 518, 518]
        
#         min_vals = channel_data.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)
#         max_vals = channel_data.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        
#         denom = max_vals - min_vals
#         denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        
#         normalized = (channel_data - min_vals) / denom
#         tensor[:, channel_idx, :, :] = normalized
#         return tensor