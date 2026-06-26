import torch
from numpy.random import normal
import  random
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import  EvalAccumulatorCuda


def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics = ['I-AUROC', 'I-AP', 'I-F1_max','P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)
        
    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()
    
    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
    
    # for i in range(torch.tensor(pr_px).size(0)):
    #     accum.add_image(torch.tensor(pr_sp[i]), torch.tensor(gt_sp[i]))
    
    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
        # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
        # hflip img
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = (augment_img[torch.randperm(augment_img.size(0))])
    return augment_img

def modify_grad(x, inds, factor=0.):
    # print(inds.shape)
    inds = inds.expand_as(x)
    # print(x.shape)
    # print(inds.shape)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

def return_3_sigma(tensor, threshold=3.0):
    mean = tensor.mean()
    std = tensor.std()
    return [mean - threshold * std, mean + threshold * std]

def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach() # 64*1*28*28
            mean_dist = point_dist.mean()  # 64
            # std_dist = point_dist.reshape(-1).std()
            # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
            factor = (point_dist/(mean_dist + 1e-8))**(y) # 64*1*28*28

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def f1_score_max(y_true, y_score, pos_label=1):
    precs, recs, thrs = precision_recall_curve(y_true, y_score, pos_label)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N

def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def save_imag_ZS(imgs, anomaly_map, gt, prototype_map, save_root, img_path):
    batch_num = imgs.shape[0]
    for i in range(batch_num):
        img_path_list = img_path[i].split('\\')
        class_name, category, idx_name = img_path_list[-4], img_path_list[-2], img_path_list[-1]
        os.makedirs(os.path.join(save_root, class_name, category), exist_ok=True)
        input_frame = denormalize(imgs[i].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_0.png'), cv2_input)
        ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_1.png'), ano_map, cmap='jet')
        gt_map = gt[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_2.png'), gt_map, cmap='gray')
        distance = prototype_map[i].view((28, 28)).cpu().detach().numpy()
        distance = cv2.resize(distance, (392, 392), interpolation=cv2.INTER_AREA)
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_3.png'), distance, cmap='jet')
        plt.close()

def overlay_similarity_map(original, sim_map, vmin=None, vmax=None):
    # Step 1:  [0, 1]
    min_val = np.min(sim_map) if vmin==None else vmin
    max_val = np.max(sim_map) if vmax==None else vmax
    if max_val - min_val < 1e-8:
        return np.zeros_like(sim_map)
    sim_map_norm = (sim_map - min_val) / (max_val - min_val)
    # sim_map_norm = sim_map
    # Step 2
    heatmap_uint8 = (sim_map_norm * 255).astype(np.uint8)

    # Step 3
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Step 4
    original_resized = cv2.resize(original, (sim_map.shape[1], sim_map.shape[0]))

    # Step 5
    overlay = cv2.addWeighted(original_resized, 0.5, colored_heatmap, 0.5, 0)

    return overlay

def save_imag_no_gt(imgs, label, anomaly_map, prototype_map, similaritys, save_root, img_path):
    batch_num = imgs.shape[0]
    patch_num = int(prototype_map.shape[1] ** 0.5)
    # for i in range(0, batch_num):
    vmin = anomaly_map.min().item()  
    vmax = anomaly_map.max().item()  
    print(vmax)
    for i in range(min(batch_num, 200)):
        img_path_list = img_path[i].split('/')
        class_name, category, idx_name = img_path_list[-4], img_path_list[-2], img_path_list[-1]
        os.makedirs(os.path.join(save_root, class_name, category, str(label[i].detach().numpy())), exist_ok=True)
        
        image_rgb = cv2.cvtColor(cv2.imread(img_path[i]), cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
        # plt.imsave(os.path.join(save_root, class_name, category, str(label[i].detach().numpy()), fr'{idx_name}.png'), image_rgb)
        ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()
        # plt.imsave(os.path.join(save_root, class_name, category, str(label[i].detach().numpy()), fr'{idx_name}_ano_map.png'), ano_map, cmap='jet', vmin=vmin, vmax=vmax)
        distance = prototype_map[i].view((patch_num, patch_num)).cpu().detach().numpy()
        distance = cv2.resize(distance, (256, 256), interpolation=cv2.INTER_CUBIC)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # anomaly map

        im1 = axes[1].imshow(overlay_similarity_map(image_rgb, ano_map))
        axes[1].set_title("Anomaly Map")
        axes[1].axis('off')

        im2 = axes[2].imshow(overlay_similarity_map(image_rgb, distance))
        # im2 = axes[2].imshow(distance, cmap='jet')
        axes[2].set_title("Prototype Distance")
        axes[2].axis('off')
        
        fig.tight_layout()
        save_dir = os.path.join(save_root, class_name, category, str(label[i].detach().numpy()))
        save_path = os.path.join(save_dir, f'{idx_name}_combined.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        num_subplots = similaritys.shape[2]  
        cols = min(num_subplots, 6)         
        rows = int(np.ceil(num_subplots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        vmin = similaritys.min().item()
        vmax = similaritys.max().item()
        for j in range(num_subplots):
            similarity = similaritys[i, :, j].view((patch_num, patch_num)).cpu().detach().numpy()
            similarity = cv2.resize(similarity, (256, 256), interpolation=cv2.INTER_CUBIC)
            ax = axes[j // cols, j % cols] if rows > 1 else axes[j]  
            im = ax.imshow(overlay_similarity_map(image_rgb, similarity, vmin, vmax), cmap='jet')
            ax.set_title(f'Similarity {j}')
            ax.axis('off')

        for j in range(num_subplots, rows * cols):
            fig.delaxes(axes[j // cols, j % cols] if rows > 1 else axes[j])
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label('Similarity Score')
        plt.savefig(os.path.join(save_root, class_name, category, str(label[i].detach().numpy()), fr'{idx_name}_similarity.png'), dpi=300)
        plt.close()
        
from sklearn.metrics import accuracy_score, f1_score, recall_score
def evaluation_batch_no_gt(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=256, pos_label=1, save_root='', ema_encoder=None, test_type='', ema_INP=True):
    model.eval()
    if ema_encoder is not None:
        ema_encoder.ema.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    
    with torch.no_grad():
        for img, label,_ , img_path in tqdm(dataloader, ncols=80):
            gt = torch.rand_like(img)
            img = img.to(device)
            output = model(img)
            if ema_encoder is not None:
                output_ema = model.forward_ema(ema_encoder.ema, img, ema_INP)
                en_ema,de_ema = output_ema[0], output_ema[1]
                en = en_ema
                de = output[1]
            else:
                en, de = output[0], output[1]
            # tsne_visualization(de)

            # g_loss = global_cosine_hm_adaptive(en, de, y=3)
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if save_root:
                save_imag_no_gt(img, label, anomaly_map, model.distance, model.similaritys, save_root, img_path)
            
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :max(1,int(anomaly_map.shape[1] * max_ratio))]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
            
            
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        
        # GPU acceleration
        # auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        aupro_px = 0
        auroc_px = 0
        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh)
        # f1 = f1_score(gt_list_sp, pr_list_sp >= thresh)
        recall = recall_score(gt_list_sp, pr_list_sp >= thresh)
        specificity = specificity_score(gt_list_sp, pr_list_sp >= thresh)

        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_px = 0
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp, pos_label=pos_label)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp, pos_label)
        f1_sp = f1_score(gt_list_sp, pr_list_sp >= thresh)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)
        f1_px = 0

    return [auroc_sp, ap_sp, f1_sp, acc, recall, specificity, thresh]

def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)
        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        
        # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

def evaluation_batch_vis_ZS(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, save_root=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            _ = model(img)
            anomaly_map = model.distance
            side = int(model.distance.shape[1]**0.5)
            anomaly_map = anomaly_map.reshape([anomaly_map.shape[0], side, side]).contiguous()
            anomaly_map = torch.unsqueeze(anomaly_map, dim=1)
            anomaly_map = F.interpolate(anomaly_map, size=img.shape[-1], mode='bilinear', align_corners=True)
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)

            save_imag_ZS(img, anomaly_map, gt, model.distance, save_root, img_path)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

         # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        
        iters = np.arange(total_iters - warmup_iters)

        try:
            self.schedule = []
            lrs_optimizer = [group['lr'] for group in optimizer.param_groups]
            for base_value in lrs_optimizer:
                warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
                schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
                self.schedule.append(np.concatenate((warmup_schedule, schedule)))
        except:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
            self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[i][self.last_epoch] for i in range(len(self.base_lrs))]    
    
from fvcore.nn import FlopCountAnalysis, parameter_count
def count_parameters_and_FLOPs(model, size=(3, 224, 224)):
    inputs = (torch.randn(1, *size),)
    flops = FlopCountAnalysis(model, inputs)
    # params = sum(p.numel() for p in model.parameters())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")
    analyze_model(model, inputs)
    return flops.total() / 1e9, params / 1e6

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_used_parameters(model, example_input):
    used_params = set()

    def forward_hook(module, input, output):
        for name, param in module.named_parameters(recurse=False):
            used_params.add(param)

    # Register hooks
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(forward_hook))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        if isinstance(example_input, (tuple, list)):
            model(*example_input)
        else:
            model(example_input)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return used_params

def analyze_model(model, example_input):
    total, trainable = count_parameters(model)
    used_params = get_used_parameters(model, example_input)
    used_param_count = sum(p.numel() for p in used_params)
    
    print(f"🔢 Total parameters:     {total/ 1e6:.2f} M")
    print(f"🟢 Trainable parameters: {trainable/ 1e6:.2f} M")
    print(f"🟠 Used in forward:      {used_param_count/ 1e6:.2f} M")
    print(f"❗ Unused parameters:     {(total - used_param_count)/ 1e6:.2f} M")