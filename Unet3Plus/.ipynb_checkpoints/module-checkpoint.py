import torch
import torch.nn as nn
import torchvision



# activation is nn.SiLU

def batch_dropout_check(use_batchnorm, use_dropout, channel, p, norm_mode, n=2):
    if !(n>0):
        raise RuntimeError(f"n should be larger than 0 but we got {n}")
    if n == 1:
        return (nn.BatchNorm2d(channel, eps=1e-03), nn.Dropout2d(inplace=True, p=p))
    else:
        batch_set = [nn.BatchNorm2d(channel, eps=1e-03) if use_batchnorm else nn.Identity() for i in range(n)]
        dropout_set = [nn.Dropout2d(inplace=True, p=p) if use_dropout else nn.Identity() for i in range(n)]
        return batch_set, dropout_set


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
        

    