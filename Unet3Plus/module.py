import torch
import torch.nn as nn
import torchvision


class SCSEModule(nn.Module):
    
    def __init__(self, in_channel, reduction=16):
        super(SCSEModule, self).__init__()
        
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//reduction, in_channel, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())
       
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)




# activation is nn.SiLU

def batch_dropout_check(use_batchnorm, use_dropout, channel, p, norm_mode, n=2):

    if norm_mode=="group":
        tmp = list()
        for i in range(1, min(channel, 32) + 1):
            if channel % i == 0:
                tmp.append(i)
        group_num = 1
        if len(tmp)>0:
            group_num = channel//tmp[-1]
        batch_set = [nn.GroupNorm(group_num, channel) if use_batchnorm else nn.Identity() for i in range(n)]
    elif norm_mode == "batch":
        batch_set = [nn.BatchNorm2d(channel, eps=1e-03) if use_batchnorm else nn.Identity() for i in range(n)]
    dropout_set = [nn.Dropout2d(inplace=True, p=p) if use_dropout else nn.Identity() for i in range(n)]
    return batch_set, dropout_set
    
class ConvNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_batchnorm , use_dropout, p, **kwargs):
        super(ConvNet, self).__init__()
        
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 3), padding = (1, 1))
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = (3, 3), padding = (1, 1))
        batch_set, dropout_set = batch_dropout_check(use_batchnorm, use_dropout, out_channels, p)
        
        self.block = nn.Sequential(conv1, batch_set[0], nn.ReLU(inplace=True), dropout_set[0]
                                    , conv2, batch_set[1], nn.ReLU(inplace=True), dropout_set[1])
        
    def forward(self, x):
        out = self.block(x)
        return out
    

class InvertedResidualNet(torchvision.models.mobilenet.InvertedResidual):
    
    def __init__(self, in_channels, out_channels, use_batchnorm , use_dropout, p, norm_mode, merge=False, **kwargs):
        super().__init__(in_channels, out_channels, stride=kwargs["stride"], expand_ratio = kwargs["expand_ratio"])
        #batch_set, dropout_set = batch_dropout_check(use_batchnorm, use_dropout, out_channels, p, n=1)
        #self.tail = nn.Sequential(batch_set[0], nn.Identity() #nn.ReLU()# because of RuntimeError occured by loss.backward() relu modification error
        #                          , dropout_set[0])

        batch_set, dropout_set = batch_dropout_check(merge, use_dropout, self.conv[0][0].out_channels, p, norm_mode, n=1)
        self.conv[0][1] = batch_set[0]
        batch_set, dropout_set = batch_dropout_check(merge, use_dropout, self.conv[1][0].out_channels, p, norm_mode, n=1)
        self.conv[1][1] = batch_set[0]
        batch_set, dropout_set = batch_dropout_check(use_batchnorm, use_dropout, self.conv[2].out_channels, p, norm_mode, n=1)
        #self.conv[3] = batch_set[0]
        self.drop = dropout_set[0]
        #batch_set, dropout_set = batch_dropout_check(use_batchnorm, use_dropout, self.conv[1].out_channels, p, norm_mode, n=1)
        #self.conv[2] = batch_set[0]
        #self.conv[0][2] = nn.SELU(inplace=True)
        #self.conv[1][2] = nn.SELU(inplace=True)
        
        #self.conv[0][2] = nn.ELU(inplace=True)
        #self.conv[1][2] = nn.ELU(inplace=True)
        
        self.conv[0][2] = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv[1][2] = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        
        
    def forward(self, x):
        
        x = self.drop(self.conv(x))
        return x
    
class ResNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_batchnorm , use_dropout, p, **kwargs):
        super(ResNet, self).__init__()
        
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 3), padding = (1, 1))
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = (3, 3), padding = (1, 1))
        
        batch_set, dropout_set = batch_dropout_check(use_batchnorm, use_dropout, out_channels, p)
        
        self.main_path = nn.Sequential(conv1, batch_set[0], nn.ReLU(inplace=True), dropout_set[0], conv2, batch_set[1])
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1))
        else:
            self.shortcut = nn.Identity()

       
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        main = self.main_path(x)
        
        
        shortcut = self.shortcut(x)
        out = self.relu(main + shortcut)
        
        
        return out
        

        
#--------------------------------------------------------------------------
# for Saliency map Attention model

class AttentionBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, middle_channel, block_level, **kwargs):
        
        super(AttentionBlock, self).__init__()

        self.featureMap_path = nn.Sequential(nn.Conv2d(in_channel, middle_channel, kernel_size=(1, 1)), nn.ReLU(inplace=True))
        self.salientMap_path = nn.Sequential(nn.MaxPool2d(kernel_size = 2**block_level), nn.Conv2d(1, middle_channel, kernel_size=(1, 1)), nn.ReLU(inplace=True))
        
        self.attentionMap_path = nn.Sequential(nn.Conv2d(middle_channel, middle_channel, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                              nn.Conv2d(middle_channel, middle_channel, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
                                              nn.Conv2d(middle_channel, 1, kernel_size=(1, 1)), nn.Sigmoid())
        
    def forward(self, x, salient_map):
        
        intermediateMap = self.featureMap_path(x) + self.salientMap_path(salient_map)
        
        AttentionMap = self.attentionMap_path(intermediateMap)
        AttentionMap = AttentionMap.expand_as(x)
        output = x * AttentionMap
        return output
        
