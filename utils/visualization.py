import cv2
import os
from utils.transforms import normalize
import numpy as np
import torch
# def visualizer(pathes, anomaly_map, img_size, save_path, cls_name):
#     for idx, path in enumerate(pathes):
#         cls = path.split('/')[-2]
#         filename = path.split('/')[-1]
#         vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
#         mask = normalize(anomaly_map[idx])
#         vis = apply_ad_scoremap(vis, mask)
#         vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
#         save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
#         if not os.path.exists(save_vis):
#             os.makedirs(save_vis)
#         cv2.imwrite(os.path.join(save_vis, filename), vis)

# def apply_ad_scoremap(image, scoremap, alpha=0.5):
#     np_image = np.asarray(image, dtype=float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

# def visualizer(path, mask,anomaly_map, img_size):
#     filename = os.path.basename(path)
#     dirname = os.path.dirname(path)
#     vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
#     mask = normalize(anomaly_map[0])
#     vis = apply_ad_scoremap(vis, mask)
#     vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
#     save_vis = os.path.join(dirname, f'anomaly_map_{filename}')
#     print(save_vis)
#     cv2.imwrite(save_vis, vis)
def draw_mask_contour(image, mask):
    """
    image: numpy H×W×3 (RGB)
    mask: numpy H×W, binary 0/1
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = image.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)  # 绿色描边
    return vis
def visualizer(img_path, gt_mask, anomaly_map, save_dir, img_size=518, data_dir=None):
    """
    img_path: 原图路径 (str)
    gt_mask: torch.Tensor [1,1,H,W] 或 numpy
    anomaly_map: torch.Tensor [1,H,W] 或 numpy
    save_dir: 保存的文件夹路径
    img_size: 输出图像大小 (默认518)
    """

    os.makedirs(save_dir, exist_ok=True)

    # 解析文件名 (去掉前面的路径，只保留 datasets 后面的部分)
    rel_path = img_path.split(data_dir+'/')[-1]   # e.g. "bottle/test/broken_small/000.png"
    rel_path = rel_path.replace("/", "-")     # e.g. "bottle-test-broken_small-000.png"
    base = rel_path.replace(".png", "")       # e.g. "bottle-test-broken_small-000"

    # 读取原图并resize (RGB)
    ori = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (img_size, img_size)), cv2.COLOR_BGR2RGB)

    # ---------- GT 可视化 ----------
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy()
    gt_mask = cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    gt_vis = draw_mask_contour(ori, gt_mask)
    save_gt = os.path.join(save_dir, f"{base}_gt.png")
    cv2.imwrite(save_gt, cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))

    # ---------- 异常图可视化 ----------
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
    if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
        anomaly_map = anomaly_map[0]
    anomaly_map = cv2.resize(anomaly_map, (img_size, img_size))
    anomaly_map = normalize(anomaly_map)
    vis = apply_ad_scoremap(ori, anomaly_map)
    save_vis = os.path.join(save_dir, f"{base}_MRAD.png")
    cv2.imwrite(save_vis, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Saved:\n {save_gt}\n {save_vis}")