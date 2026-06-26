import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def gather_loss(self, query, keys):
        B, N, C = query.shape
        _, K, _ = keys.shape
        self.similaritys = F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distribution = 1. - self.similaritys
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        similaritys = (F.cosine_similarity(keys.unsqueeze(2), keys.unsqueeze(1), dim=-1))
        # print(similaritys[0])
        if not self.training:
            counts = torch.zeros((B, K))
            for i in range(B):
                try:
                    counts[i] = torch.bincount(self.cluster_index[i], minlength=K)
                except:
                    counts[i] = torch.zeros((K,))
            print(counts.mean(dim=0))
        # balanced average: DAA loss
        one_hot_mask = F.one_hot(self.cluster_index, num_classes=K).permute(0, 2, 1).float()  # (B, K, N)
        masked_distance = one_hot_mask * self.distance.unsqueeze(1)  # (B, K, N)
        sum_distance = masked_distance.sum(dim=2)                    # (B, K)
        count = one_hot_mask.sum(dim=2) + 1e-6                       

        mean_distance = sum_distance / count                         # (B, K)

        gather_loss = mean_distance.mean(dim=1).mean()               

        return gather_loss
    
    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if (i in self.encoder_require_grad_layer) or ('all' in self.encoder_require_grad_layer):
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)

        if self.prototype_token is not None:
            agg_prototype = self.prototype_token
            for i, blk in enumerate(self.aggregation):
                agg_prototype, attn = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x, return_attention=True)
            
            g_loss = self.gather_loss(x, agg_prototype)
        else:
            agg_prototype = x
            g_loss = torch.tensor(0.0, device=x.device)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        
        return en, de, g_loss

    def forward_ema(self, ema_encoder, x, need_INP_loss=True):
        x = ema_encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(ema_encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        if self.prototype_token is not None:
            agg_prototype = self.prototype_token
            for i, blk in enumerate(self.aggregation):

                agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
            
            if need_INP_loss:
                g_loss = self.gather_loss(x, agg_prototype)
            else:
                g_loss = torch.tensor(0.0, device=x.device)
        else:
            agg_prototype = x
            g_loss = torch.tensor(0.0, device=x.device)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)









































