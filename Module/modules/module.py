import torch
import torch.nn as nn
from torchvision import models
from utils.tool import Soft_Argmax, cropping_patch, Squeeze
from layers import get_conv

class SMC_Module(nn.Module):

    def __init__(self, paper = True):
        super(SMC_Module, self).__init__()
        
        if paper:
            self.make_module()
        else:
            self.make_other_module()
            
            
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
        pose_coord = self.de_conv_soft_argmax(x)
        
        # Full-Frame cue
        full_frame = self.base_conv3(x)
        # Pose cue
        pose = self.fc_layer(pose_coord)
        
        # cropping process
        face_src, l_hand_src, r_hand_src = self.cropping_layer(crop_src, pose_coord)
        l_hand = self.hands(l_hand_src)
        r_hand = self.hands(r_hand_src)
        face = self.face(face_src)
        output = [face, torch.cat([l_hand, r_hand], dim = 0), full_frame, pose]
		
        return output
    
    def make_other_module(self):
        asd = 1
    
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
        
        self.base_conv1 = nn.Sequential(*vgg11.features[:11])
        self.base_conv2 = nn.Sequential(*vgg11.features[11:18])
        self.base_conv3 = nn.Sequential(*vgg11.features[18:], *_9th_layer, nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
        
        self.de_conv_soft_argmax = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1), nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(256, 7, kernel_size=(4, 4), stride=(2, 2), padding=1), nn.ReLU(inplace=True),
                                      Soft_Argmax())
        self.fc_layer = nn.Sequential(nn.Linear(14, 128), nn.ReLU(inplace=True), nn.Linear(128, 256), nn.ReLU(inplace=True))
        self.cropping_layer = cropping_patch()
		
		
		
		
        # weight sharing for each hands
        self.hands = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
    
        self.face = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())

        self.model = nn.ModuleList([cues["Full Frame"], cues["Pose"], cues["Face"], cues["Both Hand"]])


        
