import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import get_block


class U3Pdecoder(nn.Module):
    
    def __init__(self, encoder_channels, cat_channel, block_name, use_batchnorm, use_dropout, p, norm_mode, n_blocks=5,**kwargs):
        
        super(U3Pdecoder, self).__init__()
        
        if n_blocks != len(encoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `encoder_channels` for {} blocks.".format(
                    n_blocks, len(encoder_channels)
                )
            )
        
        
        bd_kwargs = dict(use_batchnorm=use_batchnorm, use_dropout=use_dropout, p=p, norm_mode=norm_mode)
        block_type, block_kwargs = get_block(block_name)
        blocks = dict()
        
        for decoder_layer_idx in range(n_blocks - 1, 0, -1):
            each_decoder_layer = dict()
            for encoder_layer_idx in range(1, n_blocks + 1):
                
                each_path = list()
                if encoder_layer_idx < decoder_layer_idx:
                    
                    each_path.append(nn.MaxPool2d(kernel_size = 2**(decoder_layer_idx - encoder_layer_idx)))
                    each_path.append(block_type(encoder_channels[encoder_layer_idx - 1], cat_channel, **bd_kwargs, **block_kwargs))
                
                elif encoder_layer_idx > decoder_layer_idx:
                    each_path.append(nn.Upsample(scale_factor = 2**(encoder_layer_idx - decoder_layer_idx), mode ='bilinear', align_corners=True))
                    
                    each_path.append(block_type(encoder_channels[-1] if encoder_layer_idx == n_blocks else cat_channel*n_blocks, cat_channel, **bd_kwargs, **block_kwargs))
                
                else:
                    each_path.append(block_type(encoder_channels[encoder_layer_idx - 1], cat_channel, **bd_kwargs, **block_kwargs))
                    
                each_decoder_layer[f'from_{encoder_layer_idx}_enc'] = nn.Sequential(*each_path)
            
            each_decoder_layer['fusion'] = block_type(cat_channel*n_blocks, cat_channel*n_blocks, use_batchnorm=True,
                                                      use_dropout=True, p=p, norm_mode=norm_mode, merge=False, **block_kwargs)
            blocks[f'to_{decoder_layer_idx}_dec'] = nn.ModuleDict(each_decoder_layer)
            
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleDict(blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
                
    def progressive(self, add_p):
        for each_to in self.blocks.keys():
            chosen_block = self.blocks[each_to]
            for each_from in chosen_block.keys():
                cur_block = chosen_block[each_from]
                for a in cur_block.modules():
                    if isinstance(a, nn.Dropout2d):
                        a.p += add_p
        print("decoder drop out setting clear :", self.blocks['to_1_dec']['from_1_enc'][-1].drop , " - ",self.blocks['to_1_dec']['fusion'].drop)
               
                    
    def forward(self, encoder_cache):
        
        
        decoder_cache = [encoder_cache[-1]]
        
        for decoder_layer_idx in range(self.n_blocks - 1, 0, -1):
            cat_src = []
            sub_block = self.blocks[f'to_{decoder_layer_idx}_dec']
            for i, x in enumerate(encoder_cache[:decoder_layer_idx] + decoder_cache):
                cat_src.append(sub_block[f'from_{i+1}_enc'](x))
            decoder_cache.insert(0, sub_block['fusion'](torch.cat(cat_src, dim=1)))
        return decoder_cache

    
class deepSupCGM(nn.Module):
    
    def __init__(self, n_classes, bottom_layer_channel, cat_channel, n_blocks = 5):
        super(deepSupCGM, self).__init__()
        
        sup = [nn.Sequential(nn.Conv2d(cat_channel*n_blocks, n_classes, kernel_size = (3, 3), padding=(1, 1)))]#, nn.Sigmoid())]

        for i in range(1, n_blocks):
            each_sup = nn.Sequential(
                nn.Upsample(scale_factor = 2**i, mode = "bilinear", align_corners=True), # version1
                nn.Conv2d(bottom_layer_channel if i == n_blocks-1 else cat_channel*n_blocks, n_classes, kernel_size = (3, 3), padding=(1, 1))
            )
            sup.append(each_sup)
        """
        self.scaler = nn.Sigmoid()
        self.cgm_cls = nn.Sequential( nn.Dropout(p=0.5), nn.Conv2d(bottom_layer_channel, 2, 1),
                    nn.AdaptiveMaxPool2d(1), nn.Sigmoid())
        """
        self.sup = nn.ModuleList(sup)
        self.act = nn.Softmax(dim=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
    def dotProduct(self, x, cgm_cls):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        out = torch.einsum("ijk,ij->ijk", [x, cgm_cls.expand(x.size()[:2])])
        out = out.view(B, C, H, W)
        return out
    
    def forward(self, decoder_cache):
        """
        classifier = self.cgm_cls(decoder_cache[-1]).squeeze(-1).squeeze(-1)  # (B,N,1,1)->(B,N)
        classifier = classifier.argmax(dim=1)
        classifier = classifier[:, np.newaxis].float()
        
        out = []
        for each_cache, each_module in zip(decoder_cache, self.sup):
            each_out = each_module(each_cache)
            each_out = self.dotProduct(each_out, classifier)
            each_out = self.scaler(each_out)
            out.append(each_out)
        
        out = sum(out)/len(out)
        """
        pred = list()
        
        for each_cache, each_module in zip(decoder_cache, self.sup):
            each_out = each_module(each_cache)
            pred.append(each_out)
        pred = self.act(torch.mean(torch.stack(pred, dim=0), dim=0))
        
        tmp = list()
        for piece in pred.argmax(1):
            ele, cnt = piece.unique(return_counts=True, sorted=True)
            if len(ele)<3:
                tmp.append(ele[-1])
            else:
                tmp.append(ele[1:][cnt[1:].argmax()])
        cls_token = torch.Tensor(tmp)
        cls_pred = pred.clone().argmax(1).detach()
        for piece, c in zip(cls_pred, cls_token):
            if c == 2:
                piece[piece==1] = 0
            elif c == 1:
                piece[piece==2] = 0
            elif c == 0:
                piece[piece==1] = 0
                piece[piece==2] = 0
        return pred, cls_pred, cls_token