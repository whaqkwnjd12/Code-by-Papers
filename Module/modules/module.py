import torch
import torch.nn as nn
from torchvision import models
from utils.tool import Soft_Argmax, cropping_patch, Squeeze


class SMC_Module(nn.Module):

    def __init__(self):
        super(SMC_Module, self).__init__()
        self.make_module()
    
    def forward(self, x):
        fu = self.model[0]
        po = self.model[1]
        ha = self.model[2]
        fa = self.model[3]

        tmp = []
        
        x = fu[0](x)
        crop_src = x
        crop_H, crop_W = x.size()[-2:]
        x = fu[1](x)
        soft_argmax_src = x
        tmp.append(fu[2](x))

        r, c, B= po[0](x)
        x = torch.cat((r, c), dim=-1).view(B, -1)
                                                          
        tmp.append(po[1](x))
        self.Face_src, L_Hand_src, R_Hand_src = po[2](crop_src, r, c, crop_H-1, crop_W-1)
        
        tmp.append(fa(self.Face_src))
        tmp.append(torch.cat((ha(L_Hand_src), ha(R_Hand_src)), dim=-1))
        
        output = torch.cat(tmp, dim=1)
        
        return output

    def make_module(self):
        """
            it makes modules used in SMC module.
        """
        # dict gathering different cues
        cues = {}

        # backbone network
        vgg11 = models.vgg11(pretrained = True)
        # last 9-th convolutional layer is added
        _9th_layer = [nn.Conv2d(512,512, kernel_size=(3, 3)), nn.ReLU(inplace=True)]
        # Full-Frame cues are added.
        cues["Full Frame"] = nn.ModuleList([
            # Conv 1-4
            nn.Sequential(*vgg11.features[:11]),
            # Conv 5-7
            nn.Sequential(*vgg11.features[11:18]),
            # Conv 8-9
            nn.Sequential(*vgg11.features[18:], *_9th_layer, nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())])
        
        
        cues["Pose"] = nn.ModuleList([
            # DeConv-Soft_Argmax
                                     nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1), nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(256, 7, kernel_size=(4, 4), stride=(2, 2), padding=1), nn.ReLU(inplace=True),
                                      Soft_Argmax()),
            # extra FC
                                     nn.Sequential(nn.Linear(14, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, 256), nn.ReLU(inplace=True)),
            # cropping
                                     cropping_patch()])
        
        # weight sharing for each hands
        cues["Both Hand"] = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
    
        cues["Face"] = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())

        self.model = nn.ModuleList([cues["Full Frame"], cues["Pose"], cues["Face"], cues["Both Hand"]])


        
