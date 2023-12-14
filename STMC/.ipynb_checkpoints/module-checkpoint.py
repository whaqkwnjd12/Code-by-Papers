import torch
import torch.nn as nn
from torchvision import models
from utils.tool import Soft_Argmax, cropping_patch, Squeeze

class SMC_Module(nn.Module):

    def __init__(self):
        super(SMC_Module, self).__init__()
        """
            it makes modules used in SMC module.
        """
        # dict gathering diff+erent cues
        cues = {}

        # backbone network
        vgg11 = models.vgg11(pretrained = True)
        
        # last 9-th convolutional layer for Full-Frame cues is created
        _9th_layer = [nn.Conv2d(512,512, kernel_size=(3, 3)), nn.ReLU(inplace=True)]
        
        # layer for Full-Frame cues
        self.conv1_4 = nn.Sequential(*vgg11.features[:11])
        self.conv5_7 = nn.Sequential(*vgg11.features[11:18])
        self.conv8_9 = nn.Sequential(*vgg11.features[18:], *_9th_layer, nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
        
        # layer for Pose cues
        self.de_conv_soft_argmax = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1), nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(256, 7, kernel_size=(4, 4), stride=(2, 2), padding=1), nn.ReLU(inplace=True),
                                      Soft_Argmax())
        self.fc_layer = nn.Sequential(nn.Linear(14, 128), nn.ReLU(inplace=True), nn.Linear(128, 256), nn.ReLU(inplace=True))
        
        
        # cropping layer for catching other cues (Face, L-Hand, R-Hand)
        self.cropping_layer = cropping_patch()	
        
        # R-Hand and L-Hand cues layer used weight sharing
        self.hands = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
        
        # Face cues
        self.face = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)), Squeeze())
            
            
    def forward(self, x):
        """
            parameters: 
                x        : torch.Tensor size of T x 3 X H X W
                    input
        	return:
                output   : list()
                    [face, Hands(left, right), Full-Frame, Pose]
            
            
            face		: T X 256
            both-hands	: T X 512
            full-frame	: T X 512
            pose 		: T X 256
            
        """
        
        x = self.conv1_4(x)
        
        # feature map for cropping Hands and Face cues receptive fields
        crop_src = x
        
        x = self.conv5_7(x)
        
        # pose coordinate values
        pose_coord = self.de_conv_soft_argmax(x)
        
        # Full-Frame cue
        full_frame = self.conv8_9(x)
        
        # Pose cue
        pose = self.fc_layer(pose_coord)
        
        # cropping process
        face_src, l_hand_src, r_hand_src = self.cropping_layer(crop_src, pose_coord)
        l_hand = self.hands(l_hand_src)
        r_hand = self.hands(r_hand_src)
        
        # Face cue
        face = self.face(face_src)
        
        
        output = [face, torch.cat([l_hand, r_hand], dim = -1), full_frame, pose]
		
        return output


#--------------------------------------------------------------------------------------------------------------------------------


    




