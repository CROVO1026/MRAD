from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.mlp import average_neighbor
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans



def build_cache_model(load_cache = False,  clip_model = None, train_loader_cache = None,device = None,dir=None):
    cache_dir = dir
    if load_cache == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            train_features = []
            train_labels = []
            for items in tqdm(train_loader_cache):
                images = items['img'].to(device)
                labels =  items['anomaly'].to(device)
                image_features,_ ,_,_= clip_model.encode_image(images,[6, 12, 18, 24],DPAM_layer=24)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                train_features.append(image_features)
                train_labels.append(labels)
            cache_keys = torch.cat(train_features, dim=0)
            raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
            cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)
        cache_dict = {
            "keys": cache_keys,
            "values": cache_values
        }

        torch.save(cache_dict, cache_dir)

    else:
        cache_dict = torch.load(cache_dir)
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
    return cache_keys, cache_values


def build_patch_cache_model(load_cache = False,  clip_model = None, train_loader_cache = None,device = None,dir=None):
    cache_dir = dir
    if load_cache == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            train_features = []
            train_labels = []
            for items in tqdm(train_loader_cache):
                images = items['img'].to(device)
                labels =  items['anomaly'].to(device)# b
                gt = items['img_mask'].squeeze().to(device) # b 518 518
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                image_fe,patch_features ,_,patch_projections = clip_model.encode_image(images,[6, 12, 18, 24],DPAM_layer=24)
                patch_feature = patch_features[3]
                patch_feature = average_neighbor(patch_feature)
                # patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True) #b 1369 1024
                #
                gt_resized = F.interpolate(gt.unsqueeze(1), size=(37, 37), mode='bilinear', align_corners=False)
                gt_resized = gt_resized.squeeze(1)  # (B, 37, 37)

                for i in range(images.size(0)):
                    patch = patch_feature[i]              # (1369, 768)
                    patch = patch.view(37, 37, -1)        # (37, 37, 768)
                    mask = gt_resized[i]                 # (37, 37)

                    pos_mask = (mask == 1)               # abnormal
                    neg_mask = (mask == 0)               # normal

                    if pos_mask.sum() > 0:
                        pos_feat = patch[pos_mask]    # (n_pos, 768)
                        pos_feat = pos_feat.mean(dim = 0, keepdim=True)
                        pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
                        train_features.append(pos_feat)  # anomaly → 1
                        train_labels.append(torch.tensor([1], device=device))  # anomaly → 1

                    if neg_mask.sum() > 0:
                        neg_feat = patch[neg_mask]      # (n_neg, 768)
                        neg_feat = neg_feat.mean(dim = 0, keepdim=True)
                        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
                        train_features.append(neg_feat)  # normal → 0
                        train_labels.append(torch.tensor([0], device=device))  # normal → 0

            cache_keys = torch.cat(train_features, dim=0)
            raw_labels = torch.cat(train_labels, dim=0).to(torch.int64)
            cache_values = F.one_hot(raw_labels, num_classes=2).float().to(device)
        cache_dict = {
            "keys": cache_keys,
            "values": cache_values
        }

        torch.save(cache_dict, cache_dir)

    else:
        cache_dict = torch.load(cache_dir)
        cache_keys = cache_dict["keys"].to(device)
        cache_values = cache_dict["values"].to(device)
    return cache_keys, cache_values
def compute_socre(image_features, cache_keys, cache_values, device, proj=None, need_mask=False, is_train=False, use_proj=True):
    # scale = 768**-0.5
    ori_sim_weights = torch.matmul(image_features, cache_keys.to(device).t())#b n
    loss_keys = torch.tensor(0.0, device=device)

    if use_proj and proj is not None:
        image_features_proj, cache_keys_proj = proj(image_features, cache_keys)
        sim_weights = torch.matmul(image_features_proj, cache_keys_proj.to(device).t())#b n
    else:
        sim_weights = ori_sim_weights

    if need_mask:
        th = torch.quantile(ori_sim_weights, 0.95, dim=-1, keepdim=True)
        mask = ori_sim_weights>th    #default 0.9
        mask_counts = mask.sum(dim=1)
        # print(mask_counts) 
        # print(mask.nonzero())
        sim_weights = sim_weights.masked_fill(mask, float('-inf'))
    sim_weights = F.softmax(sim_weights, dim=-1)
    # sim_weights = 0.005*torch.exp((sim_weights-1))
    logits = torch.matmul(sim_weights, cache_values.to(device).float())
    return logits, loss_keys
def compute_patch_socre(patch_features, cache_keys, cache_values, ori_sim_weights=None,
        device=None, proj=None, need_mask=False, patch_projection=False, gt_mask=None,
        anomaly_threshold=0.5, is_mradft=False, use_proj=True):

    ori_sim_weights = torch.matmul(patch_features, cache_keys.to(device).t())#b 1369 n

    if use_proj and proj is not None:
        patch_features_proj = proj(patch_features, 0)
        cache_keys_proj = proj(cache_keys, 1)
        sim_weights = torch.matmul(patch_features_proj, cache_keys_proj.T.to(device))# b 1369 n
    else:
        sim_weights = ori_sim_weights

    finetune_sim_weights = sim_weights.clone()
    if need_mask:
        th = torch.quantile(ori_sim_weights, 0.8, dim=-1, keepdim=True)
        mask = ori_sim_weights > th#test mvtec     when test visa be setted  0.85 memclip be setted 0.95
        mask_counts = mask.sum(dim=1)
        # print(mask_counts)
        # mask = mask.unsqueeze(1).expand(-1, patch_features.size(1), -1)
        sim_weights = sim_weights.masked_fill(mask, float('-inf')) 
    # similary_sum = torch.matmul(sim_weights, cache_values.to(device).float())# b 1369 2

    sim_weights = F.softmax(sim_weights, dim=-1)
    logits = torch.matmul(sim_weights, cache_values.to(device).float())  # (b, 1369, 2)

    # anomaly_weights = logits[:, :, 1]  # (b, 1369)
    # anomaly_weights = logits.permute(0, 2, 1)  # (b,2, 1369)
    # new_weights = anomaly_weights*(anomaly_weights.softmax(dim=-1)) # (b, 2, 1369)test visa
    # new_weights = anomaly_weights*(anomaly_weights/anomaly_weights.sum(dim=-1, keepdim=True))# (b, 2, 1369)test mvtec
    # new_patch_features = torch.matmul(new_weights, patch_projection)  # (b, 2, 768)
    # new_patch_features = 0

# 判断每个 patch 是异常还是正常（1通道大于0通道 -> 异常）
    if not is_mradft:
        anomaly_probs = logits[:, :, 1]
        anomaly_threshold = torch.tensor(anomaly_threshold, device=anomaly_probs.device, dtype=anomaly_probs.dtype)
        anomaly_area = (anomaly_probs > anomaly_threshold).float()  # (B, 1369)
        normal_area  = 1.0 - anomaly_area

        # 扩展维度方便与 patch_feature 相乘
        anomaly_mask = anomaly_area.unsqueeze(-1)  # (B, 1369, 1)
        normal_mask  = normal_area.unsqueeze(-1)

        # 特征加权求和
        anomaly_feat_sum = (patch_projection * anomaly_mask).sum(dim=1)  # (B, 768)
        normal_feat_sum  = (patch_projection * normal_mask).sum(dim=1)

        # 统计区域内的 patch 数量
        anomaly_count = anomaly_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        normal_count  = normal_mask.sum(dim=1).clamp(min=1.0)

        # 区域平均特征
        anomaly_token = anomaly_feat_sum / anomaly_count  # (B, 768)
        normal_token  = normal_feat_sum / normal_count

        # 如果某个 batch 没有该区域，就置为 0 向量
        anomaly_token[anomaly_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
        normal_token[normal_mask.sum(dim=1).squeeze(-1) == 0] = 0.0

        new_patch_features = torch.stack([normal_token, anomaly_token], dim=1)  # (B, 2, 768)
    else:
        new_patch_features = 0
        # print("new_patch_features is 0")
    return logits,new_patch_features,ori_sim_weights,finetune_sim_weights

def fuse_anomaly_maps(map1, map2, threshold=0.5, weight=0.5):
    """
    融合两个异常图：将map2中>threshold的区域以指定权重加到map1对应位置
    
    参数：
        map1: 第一个异常图 (518x518 numpy数组)
        map2: 第二个异常图 (518x518 numpy数组)
        threshold: map2的阈值 (默认0.5)
        weight: 融合权重 (默认0.5)
    
    返回：
        融合后的异常图
    """
    # 1. 创建map2的掩码：大于阈值的位置为True
    mask = map2 > threshold
    
    # 2. 创建融合结果图（初始复制map1）
    fused_map = map1.clone()
    
    # 3. 在掩码区域应用加权融合
    # 公式：fused = map1*(1-weight) + map2*weight
    fused_map[mask] = (1 - weight) * map1[mask] + weight * map2[mask]
    
    return fused_map

# import torch
# import torch.nn.functional as F

def similarity_experiment(simweights, cachevalues, gt_mask):
    """
    simweights: [b, 1369, n]  每个patch与memory所有元素的相似度
    cachevalues: [n, 2]       memory的标签 one-hot
    gt_mask: [b, 518, 518]    像素级GT (0=正常, 1=异常)
    """

    b, num_patches, n = simweights.shape
    simweights = simweights.softmax(dim=-1)  # [b, 1369, n]  每个patch与memory所有元素的相似度
    # 1. 聚合相似度 -> [b, 1369, 2]
    patch_sim = simweights @ cachevalues   # [b,1369,2]

    # 2. 按memory类别数量归一化
    # num_normal = cachevalues[:,0].sum().clamp(min=1)  # 防止除0
    # num_anom   = cachevalues[:,1].sum().clamp(min=1)
    # # print(f"num_normal: {num_normal}, num_anom: {num_anom}")
    # patch_sim[:,:,0] /= num_normal
    # patch_sim[:,:,1] /= num_anom
    gt_mask[gt_mask>0.5]=1
    gt_mask[gt_mask<=0.5]=0
    # 3. 下采样GT到patch级 (37x37=1369)
    gt_37 = F.interpolate(gt_mask.float(), size=(37,37), mode="bilinear")
     # [b,37,37]
    gt_flat = gt_37.view(b, -1)               # [b,1369]

    # 4. 逐图像统计
    nn_vals, aa_vals, na_vals, an_vals = [], [], [], []

    results = []
    for i in range(b):
        normal_mask = (gt_flat[i] == 0)
        anom_mask   = (gt_flat[i] == 1)

        nn_sim, aa_sim, na_sim, an_sim = None, None, None, None

        if normal_mask.sum() > 0:
            nn_patch_vals = patch_sim[i][normal_mask,0]
            na_patch_vals = patch_sim[i][normal_mask,1]
            nn_sim = nn_patch_vals.mean().item()
            na_sim = na_patch_vals.mean().item()
            nn_vals.append(nn_patch_vals)
            na_vals.append(na_patch_vals)

        if anom_mask.sum() > 0:
            aa_patch_vals = patch_sim[i][anom_mask,1]
            an_patch_vals = patch_sim[i][anom_mask,0]
            aa_sim = aa_patch_vals.mean().item()
            an_sim = an_patch_vals.mean().item()
            aa_vals.append(aa_patch_vals)
            an_vals.append(an_patch_vals)
    return nn_vals, na_vals, an_vals, aa_vals

    # # 5. 统计整体均值
    # nn_list = [r["NN"] for r in results if r["NN"] is not None]
    # aa_list = [r["AA"] for r in results if r["AA"] is not None]
    # na_list = [r["NA"] for r in results if r["NA"] is not None]
    # an_list = [r["AN"] for r in results if r["AN"] is not None]

    # summary = {
    #     "NN_mean": torch.tensor(nn_list).mean().item() if nn_list else None,
    #     "AA_mean": torch.tensor(aa_list).mean().item() if aa_list else None,
    #     "NA_mean": torch.tensor(na_list).mean().item() if na_list else None,
    #     "AN_mean": torch.tensor(an_list).mean().item() if an_list else None,
    # }

    return results
import torch

def winclip_patch_score(
    patch_features,      # (B, 1369, D)，已做过 L2 norm
    cache_keys,          # (N, D)，已做过 L2 norm
    cache_values,        # (N, 2)，[1,0]=normal, [0,1]=anomaly
    device=None,
):
    """
    基于 WinCLIP 思路的 patch-level 检索打分：
    - 对每个 patch，分别在 normal / anomaly 记忆库中取最大相似度 a, b
    - normal score = (1 + a) / 2
    - anomaly score = (1 + b) / 2
    - 返回 logits 形状为 (B, 1369, 2)
    """
    if device is None:
        device = patch_features.device

    patch_features = patch_features.to(device)        # (B, P, D)
    cache_keys = cache_keys.to(device)               # (N, D)
    cache_values = cache_values.to(device)           # (N, 2)

    B, P, D = patch_features.shape
    N, D2 = cache_keys.shape
    assert D == D2, f"Dim mismatch: patch_features D={D}, cache_keys D={D2}"

    # 根据 cache_values 划分正常 / 异常记忆
    # 约定：normal = [1, 0]，anomaly = [0, 1]
    normal_mask = (cache_values[:, 0] > cache_values[:, 1])  # (N,)
    anomaly_mask = (cache_values[:, 1] > cache_values[:, 0]) # (N,)

    if not normal_mask.any():
        raise ValueError("No normal memories found in cache_values (no [1,0]).")
    if not anomaly_mask.any():
        raise ValueError("No anomaly memories found in cache_values (no [0,1]).")

    # 计算所有 patch 对所有记忆的相似度 (cosine，相当于点积)
    # patch_features: (B, P, D)
    # cache_keys:     (N, D)
    # sim:            (B, P, N)
    sim = torch.matmul(patch_features, cache_keys.t())

    # 在 normal 记忆上取每个 patch 的最大相似度 a
    sim_normal = sim[..., normal_mask]              # (B, P, N_normal)
    a, _ = sim_normal.max(dim=-1)                   # (B, P)

    # 在 anomaly 记忆上取每个 patch 的最大相似度 b
    sim_anomaly = sim[..., anomaly_mask]            # (B, P, N_anomaly)
    b, _ = sim_anomaly.max(dim=-1)                  # (B, P)

    # 归一化到 [0, 1]： (1 + cos) / 2
    normal_score = (1.0 + a) / 2.0                  # (B, P)
    anomaly_score = (1.0 + b) / 2.0                 # (B, P)

    # 拼成 logits: (B, P, 2)
    logits = torch.stack([normal_score, anomaly_score], dim=-1)

    return logits



def winclip_image_score(
    image_features,   # (B, D)，建议已做 L2 norm
    cache_keys,       # (N, D)，建议已做 L2 norm
    cache_values,     # (N, 2)，[1,0]=normal, [0,1]=anomaly
    device,
    proj=None,        # 为了兼容原函数签名，这里不会用到
    need_mask=False,  # 为了兼容，默认不做 mask
    is_train=False    # 为了兼容
):
    """
    WinCLIP-style image-level scoring:
    - 对每个 image feature，在 normal / anomaly 记忆库上分别取最大相似度 a, b
    - normal score = (1 + a) / 2
    - anomaly score = (1 + b) / 2
    - 返回 logits: (B, 2)，以及 loss_keys=0.0（保持接口一致）
    """
    image_features = image_features.to(device)   # (B, D)
    cache_keys = cache_keys.to(device)          # (N, D)
    cache_values = cache_values.to(device)      # (N, 2)

    B, D = image_features.shape
    N, D2 = cache_keys.shape
    assert D == D2, f"Dim mismatch: image_features D={D}, cache_keys D={D2}"

    # 根据 cache_values 划分 normal / anomaly 记忆
    # 约定：normal = [1, 0]，anomaly = [0, 1]
    normal_mask = (cache_values[:, 0] > cache_values[:, 1])   # (N,)
    anomaly_mask = (cache_values[:, 1] > cache_values[:, 0])  # (N,)

    if not normal_mask.any():
        raise ValueError("No normal memories found in cache_values (no [1,0]).")
    if not anomaly_mask.any():
        raise ValueError("No anomaly memories found in cache_values (no [0,1]).")

    # 计算 image 对所有记忆的相似度 (B, N)
    # 假设 image_features / cache_keys 已经 L2 归一化，则点积就是 cosine，相似度在 [-1, 1]
    sim = torch.matmul(image_features, cache_keys.t())  # (B, N)

    # 可选的 mask（如果你确实想复用 ori_sim_weights 阈值逻辑，也可以在这里基于 sim 做）
    if need_mask:
        # 这里沿用你原来的 0.95 分位数 mask 逻辑，但注意它现在只影响 max 的候选集合
        th = torch.quantile(sim, 0.95, dim=-1, keepdim=True)  # (B, 1)
        mask = sim > th                                       # (B, N)
        # 为了简单起见：把被 mask 掉的位置设成极小值，这样后面 max 不会选到它
        sim = sim.masked_fill(mask, float("-inf"))

    # 在 normal 记忆库中取最大相似度 a
    sim_normal = sim[:, normal_mask]              # (B, N_normal)
    a, _ = sim_normal.max(dim=-1)                 # (B,)

    # 在 anomaly 记忆库中取最大相似度 b
    sim_anomaly = sim[:, anomaly_mask]            # (B, N_anomaly)
    b, _ = sim_anomaly.max(dim=-1)                # (B,)

    # 归一化到 [0,1]
    normal_score = (1.0 + a) / 2.0                # (B,)
    anomaly_score = (1.0 + b) / 2.0               # (B,)

    # 拼成 logits: (B, 2)
    logits = torch.stack([normal_score, anomaly_score], dim=-1)  # (B, 2)

    # 和原 compute_socre 对齐，返回一个 loss_keys（这里为 0）
    loss_keys = torch.tensor(0.0, device=device)

    return logits, loss_keys
