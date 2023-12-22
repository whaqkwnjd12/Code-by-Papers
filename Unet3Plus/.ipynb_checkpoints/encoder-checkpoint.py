import timm
import torch.nn as nn
from .config import get_block
from .module import batch_dropout_check

class U3Pencoder(nn.Module):
    
    def __init__(self, in_channel, encoder_block_info, add_p, **kwargs):
        
        super(U3Pencoder, self).__init__()
        pre = False

        if encoder_block_info is None:
            """
            out_c = encoder_config[encoder_name]
            block_type, block_kwargs = get_decoder_block(block_name)
            bd_kwargs = dict(use_batchnorm=use_batchnorm, use_dropout=use_dropout, p=p)
            
            block = list()
            
            block.append(nn.Sequential(block_type(in_channels, out_c[0], **bd_kwargs, **block_kwargs)))
            for i in range(1, len(out_c)):
                block.append(nn.Sequential(nn.MaxPool2d(kernel_size = 2), block_type(out_c[i-1], out_c[i], **bd_kwargs, **block_kwargs)))
            """
        else:

            encoder_module = timm.create_model(model_name=encoder_block_info["name"], pretrained = True)

            stg_idx = encoder_config['stage_idx']

            blocks = list()
            bs, _ = batch_dropout_check(use_batch, use_dropout, encoder_config["out_channels"][0], p, norm_mode, 1)
            blocks.append(nn.Sequential(nn.Conv2d(in_channel, encoder_config["out_channels"][0], kernel_size=(3, 3), padding=(1, 1)),
                                        bs,
                                        #nn.SELU(inplace=True),
                                        #nn.ELU(inplace=True),
                                        nn.LeakyReLU(inplace=True, negative_slope=0.2),
                                        #nn.ReLU(inplace=True),
                                        #nn.ReLU6(inplace=True),
                                        encoder_module.blocks[:stg_idx[0]]))

            for i in range(len(stg_idx)-1):
                blocks.append(nn.Sequential(encoder_module.blocks[stg_idx[i]:stg_idx[i+1]], nn.Dropout2d(inplace=True, p=p) if use_dropout else nn.Identity()))

            blocks.append(nn.Sequential(encoder_module.blocks[stg_idx[-1]:], nn.Dropout2d(inplace=True, p=p) if use_dropout else nn.Identity()))

            self.blocks = nn.ModuleList(blocks)

        if use_pretrained:
            for m in self.modules():
                if isinstance(m, nn.GroupNorm):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
            nn.init.xavier_normal_(blocks[0][0].weight.data)
            nn.init.normal_(blocks[0][1].weight.data, 1.0, 0.02)
            nn.init.constant_(blocks[0][1].bias.data, 0.0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def progressive(self, add_p):
        for a in self.blocks.modules():
            if isinstance(a, nn.Dropout2d):
                a.p += add_p
        print("encoder drop out setting clear :", self.blocks[-1][-1])

    def forward(self, x):
        
        cache = list()

        for layer in self.blocks:
            x = layer(x)
            cache.append(x)
            
        return cache