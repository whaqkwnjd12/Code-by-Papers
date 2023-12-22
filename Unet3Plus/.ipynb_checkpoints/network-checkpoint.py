import torch, torchvision
import torch.nn as nn
from .encoder import U3Pencoder
from .decoder import U3Pdecoder, deepSupCGM
from .config import get_timm_block

class Unet3Plus(nn.Module):
    

    def __init__(self, in_channel, cat_channel, decoder_name, n_classes, encoder_progessive_p, decoder_progressive_p,
                 encoder_block_mode = None, decoder_block_mode = None, **kwargs):
        
        super(Unet3Plus, self).__init__()
        

        # torchvision.models.resnet34, 101, vgg19, mobilenet_v2
        if encoder_block_mode is not None:
            encoder_block_info = get_timm_block(encoder_block_mode)
        self.encoder = U3Pencoder(in_channel = in_channel,
                                  encoder_block_info = encoder_block_info,
                                  add_p = encoder_progessive_p,
                                 **kwargs)
        

        self.decoder = U3Pdecoder(encoder_channels = encoder_config['out_channels'],
                                  cat_channel = cat_channel,
                                  block_name = decoder_name,
                                  use_batchnorm = decoder_batch,
                                  use_dropout = use_dropout_d,
                                  p=decoder_p,
                                  norm_mode = norm_mode,
                                 **kwargs)
        
        self.sup = deepSupCGM(n_classes = n_classes,
                              bottom_layer_channel = encoder_config['out_channels'][-1],
                              cat_channel = cat_channel)

        
    def progressive(self):
        
        if self.use_d_en:
            self.encoder.progressive(self.pc*2/3)
        if self.use_d_de:
            self.decoder.progressive(self.pc)
        
        
    def forward(self, x):
  

        encoder_cache = self.encoder(x)
        
        
        decoder_cache = self.decoder(encoder_cache)
        
        
        pred, cls_pred, cls_token = self.sup(decoder_cache)
        
        return pred, cls_pred, cls_token