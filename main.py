import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch_no_gt, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

# Dataset-Related Modules
from dataset import prepare_dataset
from dataset import get_data_transforms

# Model-Related Modules
from models import vit_encoder
from models.DNP_ConFormer import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')

warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)
    # Data Preparation
    data_transform, data_transforms_test, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    
    pos_label = 1
    train_data = prepare_dataset(os.path.join(args.data_path,'train'), transform = data_transform)
    valid_data = prepare_dataset(os.path.join(args.data_path,'test'), transform = data_transforms_test, need_img_path = True)
    print_fn('train image number:{}'.format(train_data.get_cls_num_list()))
    print_fn('valid image number:{}'.format(valid_data.get_cls_num_list()))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                    drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # fuse_layer_encoder = [[4, 5], [6, 7]]
    # fuse_layer_decoder = [[4, 5], [6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    if args.INP_num > 0:
        INP = nn.ParameterList(
                        [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                        for _ in range(1)])
    else:
        INP = [None]

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder, encoder_require_grad_layer = args.encoder_require_grad_layer,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    # count_parameters_and_FLOPs(model, size=(3, 252, 252))
    model = model.to(device)
    resize_mask = 256
    if args.ema_contrast:
        from timm.utils import ModelEma
        ema_encoder = ModelEma(encoder, decay=args.ema_decay)
    else:
        ema_encoder = None
    need_ema_INP = True if args.emaINP_loss_weight != 0 else False
    if args.phase == 'train':
        # Model Initialization
        if args.INP_num > 0:
            trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        else:
            trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if args.resume:
            model.load_state_dict(torch.load(args.resume), strict=True)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        # optimizer.param_groups[0]['lr'] = 0
        if len(args.encoder_require_grad_layer) != 0:
            optimizer.add_param_group({'params': encoder.parameters(), 'lr': args.encoder_lr})
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=args.lr, final_value=args.final_lr, total_iters=args.total_iters,
                                           warmup_iters=0)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        best_AUC =  [0,0,0,0,0,0,0]
        best_epoch = 0
        iters = 0
        for epoch in range(1 + args.total_iters // len(train_dataloader)):
            model.train()
            if not args.encoder_train_bn:
                model.encoder.train(False)
            loss_list = []
            g_loss_list = []
            INP_loss_list = []

            for img, _ ,_ in train_dataloader:
                iters += 1
                if iters > args.total_iters:
                    break
                img = img.to(device)
                en, de, INP_loss = model(img)
                if args.ema_contrast:
                    ema_encoder.update(model.encoder)
                    ema_en, ema_de, ema_INP_loss = model.forward_ema(ema_encoder.ema, img, need_INP_loss=need_ema_INP)
                    g_loss = global_cosine_hm_adaptive(ema_en, de, y=args.y)
                else:    
                    ema_INP_loss = torch.tensor(0.0, device=device)
                    g_loss = global_cosine_hm_adaptive(en, de, y=args.y)
                
                loss = g_loss + args.INP_loss_weight*INP_loss + args.emaINP_loss_weight*ema_INP_loss

                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                g_loss_list.append(g_loss.item())
                INP_loss_list.append(INP_loss.item())
                lr_scheduler.step()
                
                if (iters == 1) or ((iters) % 200 == 0) or (iters > args.total_iters-5):
                    print_fn('iters [{}/{}], loss:{:.4f}, g_loss:{:.4f}, INP_loss:{:.4f}'.format(iters, args.total_iters, np.mean(loss_list), np.mean(g_loss_list), np.mean(INP_loss_list)))
                    if iters >= args.total_iters/2:
                        model.eval()
                        # results = evaluation_batch_no_gt(model, valid_dataloader, device, max_ratio=args.max_ratio, resize_mask=resize_mask, pos_label=pos_label, save_root=os.path.join(args.save_dir, args.save_name, args.item, 'image_epoch'+str(epoch+1)),ema_encoder=ema_encoder, test_type=args.test_type, ema_INP=need_ema_INP)
                        results = evaluation_batch_no_gt(model, valid_dataloader, device, max_ratio=args.max_ratio, resize_mask=resize_mask, pos_label=pos_label, ema_encoder=ema_encoder, test_type=args.test_type, ema_INP=need_ema_INP)
                        auroc_sp, ap_sp, f1_sp, acc, recall, specificity, thresh = results
                        if results[0] > best_AUC[0]:
                            best_AUC = results
                            best_epoch = iters
                            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model_best.pth'))
                            if ema_encoder is not None:
                                torch.save(ema_encoder.ema.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'ema_encoder_best.pth'))
                        print_fn(
                            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, Acc:{:.4f}, Recall:{:.4f}, Specificity:{:.4f}, Thresh:{:.4f}, best_epoch:{}'.format(
                                'valid', auroc_sp, ap_sp, f1_sp, acc, recall, specificity, thresh, best_epoch))
        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model.pth'))
        if ema_encoder is not None:
            torch.save(ema_encoder.ema.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'ema_encoder.pth'))
        print_fn('best_epoch: {}, Best AUC: {:.4f}, AP: {:.4f}, F1: {:.4f}, Acc: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Thresh: {:.4f}'.format(
                        best_epoch, best_AUC[0], best_AUC[1], best_AUC[2], best_AUC[3], best_AUC[4], best_AUC[5], best_AUC[6]))
        return results, best_AUC
    elif args.phase == 'test':
        # Test
        model_path = os.path.join(args.save_dir, args.save_name, args.item, 'model.pth')
        print('Loading model from:', model_path)
        model.load_state_dict(torch.load(model_path), strict=True)

        if ema_encoder is not None:
            ema_encoder_path = os.path.join(args.save_dir, args.save_name, args.item, 'ema_encoder.pth')
            ema_encoder.ema.load_state_dict(torch.load(ema_encoder_path), strict=True)
        model.eval()
        results = evaluation_batch_no_gt(model, valid_dataloader, device, max_ratio=args.max_ratio, resize_mask=resize_mask, pos_label = pos_label, save_root=os.path.join(args.save_dir, args.save_name, args.item, 'image_test'), ema_encoder=ema_encoder, test_type=args.test_type, ema_INP=need_ema_INP)
        return results, results

import time
if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "4,5"
    gpu_ids = '0'
    print('gpu_ids:', gpu_ids)
    
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'APTOS2019') # 'APTOS2019' or 'Br35H' or 'ISIC2018' or 'OCT2017'
    parser.add_argument('--data_path', type=str, default=r'./dataset/APTOS2019')  # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='./DNP-ConFormer/')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=252)
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_iters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--final_lr', type=float, default=1e-5)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--max_ratio', type=float, default=0.01)  # max_ratio for evaluation
    parser.add_argument('--encoder_require_grad_layer', type=list, default=['all'])
    parser.add_argument('--ema_contrast', type=bool, default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--test_type', type=str, default='')
    parser.add_argument('--INP_loss_weight', type=float, default=0)
    parser.add_argument('--emaINP_loss_weight', type=float, default=0)
    parser.add_argument('--y', type=int, default=3)
    parser.add_argument('--resume', type=str, default=None)  # Path to resume model
    parser.add_argument('--encoder_train_bn', type=bool, default=True)
    parser.add_argument('--method', type=str, default='M2') # 'M0' or 'M1' or 'M2'
    args = parser.parse_args()
    
    device = 'cuda:'+gpu_ids if torch.cuda.is_available() else 'cpu'
    
    print('ema_contrast:', args.ema_contrast, 'ema_decay:', args.ema_decay, 'test_type=', args.test_type)
    args.item  = args.method+'_INP_N=' + str(args.INP_num ) + '_'+str(args.INP_loss_weight)+ '_BINPLoss'+'_'+str(args.emaINP_loss_weight)+'_BemaINPsLoss_' + str(args.total_iters)+'lr='+str(args.lr) +'final_lr_' + str(args.final_lr) + '_encoderlr='+str(args.encoder_lr) + '_batch=' + str(args.batch_size) + '_max_ratio='+str(args.max_ratio)+'_y=' + str(args.y) + '_ema_decay=' + str(args.ema_decay)

    args.save_name = args.save_dir + args.save_name + f'/_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name, args.item ))
    print_fn = logger.info
    
    result_list = []
    result_best_list = []
    for item in [0,1,2,3,4]: # run 5 times
        print_fn(args)
        results, results_best = main(args)
        auroc_sp, ap_sp, f1_sp, acc, recall, specificity, thresh = results
        best_auroc_sp, best_ap_sp, best_f1_sp, best_acc, best_recall, best_specificity, best_thresh = results_best
        result_list.append([str(item) + '_' + args.item, auroc_sp, ap_sp, f1_sp, acc, recall, specificity, thresh])
        result_best_list.append([str(item) + '_' + args.item, best_auroc_sp, best_ap_sp, best_f1_sp, best_acc, best_recall, best_specificity, best_thresh])
        torch.cuda.empty_cache()
    for last_or_best in ['last', 'best']:
        if last_or_best == 'best':
            result_list = result_best_list
        mean_auroc_sp = np.mean([result[1] for result in result_list])
        mean_ap_sp = np.mean([result[2] for result in result_list])
        mean_f1_sp = np.mean([result[3] for result in result_list])

        mean_acc = np.mean([result[4] for result in result_list])
        mean_recall = np.mean([result[5] for result in result_list])
        mean_specificity = np.mean([result[6] for result in result_list])
        mean_thresh = np.mean([result[7] for result in result_list])

        print_fn(result_list)
        print_fn(
            last_or_best + '_Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, Acc:{:.4f}, Recall:{:.4f}, Specificity:{:.4f}, Thresh:{:.4f}'.format(
                mean_auroc_sp, mean_ap_sp, mean_f1_sp,
                mean_acc, mean_recall, mean_specificity, mean_thresh))
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
