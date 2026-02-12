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
import numpy as np
import os
import random
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from mrad import build_cache_model, compute_socre, compute_patch_socre, build_patch_cache_model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = args.device if torch.cuda.is_available() else "cpu"
    model_type = args.model_type

    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    dataset_name = args.dataset
    cache_name = dataset_name

    # Initialize model components
    patch_proj = Projector(1024, 768, length=2)
    image_proj = MLP()
    patch_proj.to(device)
    image_proj.to(device)

    # mrad-clip requires additional components
    if model_type == 'mrad-clip':
        prompt_proj = AnomalyMLP()
        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
        prompt_learner.to(device)
        prompt_proj.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=24)

    # Loss functions
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # Load memory bank
    cache_keys, cache_values = build_cache_model(
        load_cache=True, clip_model=model, train_loader_cache=None,
        device=device, dir=os.path.join(args.cache_dir, f'cache_model_{dataset_name}.pt')
    )
    cache_keys_patch, cache_values_patch = build_patch_cache_model(
        load_cache=True, clip_model=model, train_loader_cache=None,
        device=device, dir=os.path.join(args.cache_dir, f'cache_patch_model_{dataset_name}.pt')
    )
    print(f"cache_key: {cache_keys.shape}")
    print(f"cache_keys_patch: {cache_keys_patch.shape}")

    # Set total epochs based on model_type
    if model_type == 'mrad-ft':
        total_epochs = args.ft_epochs  # default 1
    else:  # mrad-clip
        total_epochs = args.ft_epochs + args.clip_epochs  # default 1 + 5 = 6

    # Stage 1 optimizer: train patch_proj and image_proj (mrad-ft stage)
    optimizer_ft = torch.optim.Adam(
        list(patch_proj.parameters()) + list(image_proj.parameters()),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )
    # Cosine learning rate decay only for standalone mrad-ft training
    if model_type == 'mrad-ft':
        ft_total_steps = args.ft_epochs * len(train_dataloader)
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=ft_total_steps, eta_min=1e-6
        )

    # Stage 2 optimizer: train prompt_learner and prompt_proj (mrad-clip stage)
    if model_type == 'mrad-clip':
        optimizer_clip = torch.optim.Adam(
            list(prompt_learner.parameters()) + list(prompt_proj.parameters()),
            lr=args.learning_rate, betas=(0.5, 0.999)
        )

    for epoch in tqdm(range(total_epochs)):
        model.eval()
        patch_proj.train()
        image_proj.train()

        if model_type == 'mrad-clip':
            prompt_learner.train()
            prompt_proj.train()

        loss_list = []
        seg_loss_list = []
        clip_loss_list = []

        # Determine current stage
        is_ft_stage = (epoch < args.ft_epochs)

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features, all_cls_tokens, patch_projections = model.encode_image(
                    image, args.features_list, DPAM_layer=24
                )

            patch_pojection = patch_projections[3]
            patch_pojection = average_neighbor(patch_pojection)
            patch_pojection = patch_pojection / patch_pojection.norm(dim=-1, keepdim=True)

            patch_f_bia = patch_features[3]
            patch_f_bia = average_neighbor(patch_f_bia)
            patch_f_bia = patch_f_bia / patch_f_bia.norm(dim=-1, keepdim=True)

            # Compute segmentation logits
            seg_logit, patch_f_bia, _, _ = compute_patch_socre(
                patch_f_bia, cache_keys_patch, cache_values_patch,
                device=device, proj=patch_proj, need_mask=True,
                patch_projection=patch_pojection, use_proj=True
            )
            seg_similarity_map = AnomalyCLIP_lib.get_similarity_map(seg_logit, args.image_size).permute(0, 3, 1, 2)

            # Segmentation loss
            seg_loss = loss_focal(seg_similarity_map, gt) + loss_dice(seg_similarity_map[:, 1, :, :], gt)
            seg_loss_list.append(seg_loss.item())

            if is_ft_stage:
                # Stage 1: train patch_proj (segmentation) and image_proj (classification)
                # Classification loss
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits, _ = compute_socre(image_features, cache_keys, cache_values, device, proj=image_proj, need_mask=True, is_train=False)
                logits = logits / 0.07
                image_loss = F.cross_entropy(logits.squeeze(), label.long().to(device))

                # Total loss = segmentation loss + classification loss
                total_loss = seg_loss + image_loss

                optimizer_ft.zero_grad()
                total_loss.backward()
                optimizer_ft.step()
                if model_type == 'mrad-ft':
                    scheduler_ft.step()
                loss_list.append(total_loss.item())
            else:
                # Stage 2: train prompt_learner and prompt_proj (mrad-clip)
                # Freeze patch_proj
                for param in patch_proj.parameters():
                    param.requires_grad = False

                bias = prompt_proj(patch_f_bia[:, 0, :], patch_f_bia[:, 1, :])
                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute CLIP alignment loss
                similarity_map_list = []
                for idx, patch_feature in enumerate(patch_projections):
                    if idx >= args.feature_map_layer[3]:
                        patch_feature = average_neighbor(patch_feature)
                        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                        similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity, args.image_size).permute(0, 3, 1, 2)
                        similarity_map_list.append(similarity_map)

                clip_loss = 0
                for sim_map in similarity_map_list:
                    clip_loss += loss_focal(sim_map, gt)
                    clip_loss += loss_dice(sim_map[:, 1, :, :], gt)
                    clip_loss += loss_dice(sim_map[:, 0, :, :], 1 - gt)

                clip_loss_list.append(clip_loss.item())

                optimizer_clip.zero_grad()
                clip_loss.backward()
                optimizer_clip.step()
                loss_list.append(clip_loss.item())

        # Logging
        if (epoch + 1) % args.print_freq == 0:
            if is_ft_stage:
                if model_type == 'mrad-ft':
                    logger.info('epoch [{}/{}] [FT Stage], seg_loss: {:.4f}, total_loss: {:.4f}, lr: {:.6f}'.format(
                        epoch + 1, total_epochs, np.mean(seg_loss_list), np.mean(loss_list),
                        optimizer_ft.param_groups[0]['lr']))
                else:
                    logger.info('epoch [{}/{}] [FT Stage], seg_loss: {:.4f}, total_loss: {:.4f}'.format(
                        epoch + 1, total_epochs, np.mean(seg_loss_list), np.mean(loss_list)))
            else:
                logger.info('epoch [{}/{}] [CLIP Stage], clip_loss: {:.4f}'.format(
                    epoch + 1, total_epochs, np.mean(clip_loss_list)))

        # Save model
        if (epoch + 1) % args.save_freq == 0:
            if model_type == 'mrad-ft':
                ckp_path = os.path.join(args.save_path, f'mrad_ft_epoch_{epoch + 1}.pth')
                torch.save({
                    "image_proj": image_proj.state_dict(),
                    "patch_proj": patch_proj.state_dict()
                }, ckp_path)
            else:  # mrad-clip
                ckp_path = os.path.join(args.save_path, f'mrad_clip_epoch_{epoch + 1}.pth')
                torch.save({
                    "prompt_learner": prompt_learner.state_dict(),
                    "prompt_proj": prompt_proj.state_dict(),
                    "image_proj": image_proj.state_dict(),
                    "patch_proj": patch_proj.state_dict()
                }, ckp_path)

    # Final save
    if model_type == 'mrad-ft':
        final_path = os.path.join(args.save_path, 'mrad_ft_final.pth')
        torch.save({
            "image_proj": image_proj.state_dict(),
            "patch_proj": patch_proj.state_dict()
        }, final_path)
        logger.info(f'MRAD-FT model saved to {final_path}')
    else:
        final_path = os.path.join(args.save_path, 'mrad_clip_final.pth')
        torch.save({
            "prompt_learner": prompt_learner.state_dict(),
            "prompt_proj": prompt_proj.state_dict(),
            "image_proj": image_proj.state_dict(),
            "patch_proj": patch_proj.state_dict()
        }, final_path)
        logger.info(f'MRAD-CLIP model saved to {final_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MRAD Training", add_help=True)
    # Data paths
    parser.add_argument("--data_path", type=str, default="./data/VISA", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/released', help='path to save results')
    parser.add_argument("--cache_dir", type=str, default='./cache', help='directory for cache files')
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")

    # Model parameters
    parser.add_argument("--depth", type=int, default=9, help="learnable_text_embedding_depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="Prompt_length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable_text_embedding_length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="feature map layers")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # Training parameters
    parser.add_argument("--model_type", type=str, default='mrad-clip', choices=['mrad-ft', 'mrad-clip'],
                        help='Model type to train: mrad-ft or mrad-clip')
    parser.add_argument("--ft_epochs", type=int, default=1, help="epochs for FT stage (patch_proj training)")
    parser.add_argument("--clip_epochs", type=int, default=5, help="epochs for CLIP stage (prompt_learner + prompt_proj training)")
    parser.add_argument("--learning_rate", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:0')

    args = parser.parse_args()
    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    train(args)
