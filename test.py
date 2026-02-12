import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from models.prompt_learner import AnomalyCLIP_PromptLearner
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.transforms import normalize, get_transform
from utils.dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
from models.mlp import AnomalyMLP,MLP,Projector,average_neighbor
from models.attention import NormalFeatureAttention,CrossAttentionPooling
import os
import time
import random
import numpy as np
from tabulate import tabulate
from mrad import build_cache_model,compute_socre,compute_patch_socre,build_patch_cache_model,similarity_experiment
from utils.visualization import visualizer
from utils.metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    k = args.k
    if dataset_name == 'mvtec':
        seg_classi = 'mvtec'
        cache_name = 'visa'
    elif dataset_name == 'visa':
        cache_name = 'mvtec'
        seg_classi = 'visa'
    else:
        cache_name ='visa'
        seg_classi = 'mvtec'

    logger = get_logger(args.save_path)
    device = args.device

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()
  

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    # cache_key, cache_value = build_cache_model(load_cache=False, clip_model=model, train_loader_cache=test_dataloader, device=device,dir='/root/data/xcr3/projects/AnomalyCLIP-main/cache_model_visa.pt')
    # mvtec_data = Dataset(root='/root/data/xcr3/datasets/MVTEC', transform=preprocess, target_transform=target_transform, dataset_name = 'mvtec')
    # mvtec_dataloader = torch.utils.data.DataLoader(mvtec_data, batch_size=1, shuffle=True)
    # cache_key, cache_value = build_cache_model(load_cache=False, clip_model=model, train_loader_cache=mvtec_dataloader, device=device,dir='/root/data/xcr3/projects/AnomalyCLIP-main/cache_model_mvtec.pt')

    obj_list = test_data.obj_list

    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['pixel-ap'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    image_proj = MLP()
    patch_proj = Projector(1024,768,length=2)
    prompt_proj = AnomalyMLP()
    normal_atten = CrossAttentionPooling()

    model_type = args.model_type

    # Only mrad-clip and mrad-ft need to load weights
    if model_type in ['mrad-clip', 'mrad-ft']:
        checkpoint = torch.load(args.checkpoint_path)
        image_proj.load_state_dict(checkpoint["image_proj"])
        image_proj.to(device)
        patch_proj.load_state_dict(checkpoint["patch_proj"])
        patch_proj.to(device)

    # Only mrad-clip needs prompt_learner and prompt_proj
    if model_type == 'mrad-clip':
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(device)
        prompt_proj.load_state_dict(checkpoint["prompt_proj"])
        prompt_proj.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 24)

    cache_key, cache_value = build_cache_model(load_cache=True, clip_model=model, train_loader_cache=None, device=device, dir=os.path.join(args.cache_dir, f'cache_model_{cache_name}.pt'))
    cache_keys_patch, cache_values_patch = build_patch_cache_model(load_cache=True, clip_model=model, train_loader_cache=None, device=device, dir=os.path.join(args.cache_dir, f'cache_patch_model_{cache_name}.pt'))
    print(f"cache_key:{cache_key.shape}")
    print(f"cache_keys_patch:{cache_keys_patch.shape}")


    start = time.time()
    norm_num,anom_num = 0,0
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        if items['anomaly'] == 0:
            norm_num += 1
        else:
            anom_num += 1

        with torch.no_grad():
            image_features, patch_features,all_cls_tokens,patch_projections = model.encode_image(image, features_list,DPAM_layer=24)

            patch_pojection = patch_projections[3]
            patch_pojection = average_neighbor(patch_pojection)
            patch_pojection = patch_pojection / patch_pojection.norm(dim=-1, keepdim=True)
            patch_f_bia = patch_features[3]
            patch_f_bia = average_neighbor(patch_f_bia)
            patch_f_bia = patch_f_bia / patch_f_bia.norm(dim=-1, keepdim=True)

            # Decide whether to use projection
            use_proj = (model_type != 'mrad-tf')

            seg_logit, patch_f_bia, ori_weight, ft_weight = compute_patch_socre(
                patch_f_bia, cache_keys_patch, cache_values_patch,
                device=device,
                proj=patch_proj if use_proj else None,
                need_mask=False,
                patch_projection=patch_pojection,
                gt_mask=items['img_mask'],
                is_mradft=(model_type != 'mrad-clip'),
                use_proj=use_proj
            )
            seg_similarity_map = AnomalyCLIP_lib.get_similarity_map(seg_logit, args.image_size)

            # CLIP image-text alignment (only needed for mrad-clip)
            if model_type == 'mrad-clip':
                bias = prompt_proj(patch_f_bia[:,0,:], patch_f_bia[:,1,:])
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cache_logits, _ = compute_socre(
                image_features, cache_key, cache_value, device,
                proj=image_proj if use_proj else None,
                use_proj=use_proj
            )

            text_probs = cache_logits
            text_probs = text_probs[:, 1]
            anomaly_map_list = []

            for idx, patch_feature in enumerate(patch_projections):
                if idx >= args.feature_map_layer[3]:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                    if model_type == 'mrad-clip':
                        # mrad-clip uses CLIP image-text alignment anomaly_map
                        similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, args.image_size)
                        anomaly_map1 = (similarity_map[...,1] + 1 - similarity_map[...,0]) / 2.0
                        anomaly_map_list.append(anomaly_map1)
                    else:
                        # mrad-ft and mrad-tf use seg_similarity_map
                        anomaly_map2 = seg_similarity_map[...,1]
                        anomaly_map_list.append(anomaly_map2)


            anomaly_map = torch.stack(anomaly_map_list)#[1 518 518]
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            
            anomaly_map = anomaly_map.sum(dim = 0)
            map_flat = anomaly_map.flatten()
            topk_values,_ = torch.topk(map_flat, int(map_flat.shape[0]*0.01))
            topk_mean = topk_values.mean()
            text_probs = (1-k)*topk_mean + k*text_probs
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            # text_probs = (1-k)*anomaly_map.max() + k*text_probsS
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
        if args.visulize_bool:
            visualizer(items['img_path'][0], gt_mask,anomaly_map.detach().cpu().numpy(),save_dir=os.path.join(args.save_path, 'visualization', dataset_name), img_size=args.image_size,data_dir = dataset_dir)
    end = time.time()
    print(f"time cost:{end-start}")
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    compute_pixel_aupro = args.compute_pixel_aupro
    pixel_aupro_list = [] if compute_pixel_aupro else None
    pixel_f1_list = []
    for obj in tqdm(obj_list):
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            if compute_pixel_aupro:
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))         
            pixel_auroc_list.append(pixel_auroc)
            if compute_pixel_aupro:
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                table.append(str(np.round(pixel_aupro * 100, decimals=1)))
                pixel_aupro_list.append(pixel_aupro)
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_ap_list.append(image_ap) 

        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        mean_row = ['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1))]
        headers = ['objects', 'pixel_auroc']
        if compute_pixel_aupro and pixel_aupro_list:
            mean_row.append(str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)))
            headers.append('pixel_aupro')
        table_ls.append(mean_row)
        results = tabulate(table_ls, headers=headers, tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        mean_row = ['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1))]
        headers = ['objects', 'pixel_auroc']
        if compute_pixel_aupro and pixel_aupro_list:
            mean_row.append(str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)))
            headers.append('pixel_aupro')
        mean_row.extend([
            str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
            str(np.round(np.mean(image_ap_list) * 100, decimals=1))
        ])
        headers.extend(['image_auroc', 'image_ap'])
        table_ls.append(mean_row)
        results = tabulate(table_ls, headers=headers, tablefmt="pipe")
    logger.info("\n%s", results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MRAD Testing", add_help=True)
    # Paths
    parser.add_argument("--data_path", type=str, default="./data/MVTEC", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/test_on_mvtec', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str,
        default='./checkpoints/test_on_mvtec.pth', help='path to checkpoint')
    parser.add_argument("--cache_dir", type=str, default='./cache', help='directory for cache files')
    # Model parameters
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="learnable_text_embedding_depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable_text_embedding_length")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="feature map layers")
    parser.add_argument("--metrics", type=str, default='image-pixel-level', help='metrics: image-level, pixel-level, image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--k", type=float, default=0.7, help="fusion weight (0.5, 0.7, 0.8)")
    parser.add_argument("--device",type=str,default='cuda:1')
    parser.add_argument("--visulize_bool",type=bool,default=False)
    parser.add_argument("--compute_pixel_aupro", type=bool, default=False, help="compute pixel-level AUPRO metric in addition to AUROC")
    parser.add_argument("--model_type", type=str, default='mrad-clip',
        choices=['mrad-clip', 'mrad-ft', 'mrad-tf'],
        help='Model type: mrad-clip (full), mrad-ft (fine-tuned), mrad-tf (train-free)')
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
