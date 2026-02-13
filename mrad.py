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


    if not is_mradft:
        anomaly_probs = logits[:, :, 1]
        anomaly_threshold = torch.tensor(anomaly_threshold, device=anomaly_probs.device, dtype=anomaly_probs.dtype)
        anomaly_area = (anomaly_probs > anomaly_threshold).float()  # (B, 1369)
        normal_area  = 1.0 - anomaly_area

     
        anomaly_mask = anomaly_area.unsqueeze(-1)  # (B, 1369, 1)
        normal_mask  = normal_area.unsqueeze(-1)

 
        anomaly_feat_sum = (patch_projection * anomaly_mask).sum(dim=1)  # (B, 768)
        normal_feat_sum  = (patch_projection * normal_mask).sum(dim=1)


        anomaly_count = anomaly_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        normal_count  = normal_mask.sum(dim=1).clamp(min=1.0)


        anomaly_token = anomaly_feat_sum / anomaly_count  # (B, 768)
        normal_token  = normal_feat_sum / normal_count


        anomaly_token[anomaly_mask.sum(dim=1).squeeze(-1) == 0] = 0.0
        normal_token[normal_mask.sum(dim=1).squeeze(-1) == 0] = 0.0

        new_patch_features = torch.stack([normal_token, anomaly_token], dim=1)  # (B, 2, 768)
    else:
        new_patch_features = 0
        # print("new_patch_features is 0")
    return logits,new_patch_features,ori_sim_weights,finetune_sim_weights

