import torch
import torch.nn as nn
from torchvision import models
from utils.tool import Soft_Argmax, cropping_patch, Squeeze
from layers import get_conv

class SMC_Module(nn.Module):

    def __init__(self, kwargs):
        super(SMC_Module, self).__init__()
        self.make_module()
        
        if kwargs['paper_code']:
            self.make_module()
        else:
            self.base_conv1 = get_layer(kwargs["base_conv1"])
            self.base_conv1 = get_layer(kwargs["base_conv2"])
            self.base_conv1 = get_layer(kwargs["base_conv3"])
            
            self.fase_conv = get_layer(kwargs["Face_conv"])
            self.hands_conv = get_layer(kwargs["Hand_conv"])
            
            self.de_conv = get_layer(kwargs["De_conv"])
            self.soft_argmax = Soft_Argmax()
            
            self.fc_layer = get_layer(kwargs["Affine"])
            
            self.cropping_layer = cropping_patch()
            
            
    def forward(self, x):
        """
            parameters: torch.Tensor size of T x 3 X H X W
                input
        	return full-frame, pose, face, both-hands
            
            
            full-frame	: T X 512
            pose 		: T X 256
            face		: T X 256
            both-hands	: T X 512
        """
        
        x = self.base_conv1(x)
        
        # tmp value for cropping Hands and Face features
        crop_src = x
        
        x = self.base_conv2(x)
        
        # pose coordinate values
        pose_coord = self.soft_argmax(self.de_conv(x))
        
        # Full-Frame cue
        full_frame = self.base_conv3(x)
        # Pose cue
        pose_cue = self.fc_layer(pose_coord)
        
        # cropping process
        face_src, l_hand_src, r_hand_src = self.cropping_layer(crop_src, pose_coord)
        
        
        return output
    
    def make_other_type(kwargs):
        self.base_conv1 = get_layer(kwargs["base_conv1"])
        self.base_conv1 = get_layer(kwargs["base_conv2"])
        self.base_conv1 = get_layer(kwargs["base_conv3"])
        
        self.fase_conv = get_layer(kwargs["Face_conv"])
        
        self.hands_conv = get_layer(kwargs["Hand_conv"])
        
        self.de_conv = get_layer(kwargs["De_conv"])
        self.soft_argmax = Soft_Argmax()
            
        self.fc_layer = get_layer(kwargs["Affine"])

        self.cropping_layer = cropping_patch()
        
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
        
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


        
