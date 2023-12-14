import torch
import torch.nn as nn

# TMC module

class TMC_Module(nn.Module):
    
    def __init__(self, in_channels, out_channel, kernel_size=5, padding=2, stride=1):
        """
            we use two TMB block
            
            parameters:
                in_channels : list()
                    list of channels of each feature cues
                out_channel: torch.Tensor size of 1 dimension
                    sum of out_channels from multi cues
        """
        super(TMC_Module, self).__init__()
        
        # C = num of cues
        C = len(in_channels)
        out_channels = list()
        tp1_intra = list()
        tp2_intra = list()
        for _ in range(C):
            out_channels.append(out_channel//C)
            tp1_intra.append(nn.MaxPool1d(kernel_size=2, stride=2))
            tp2_intra.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.TMC_block1 = TMC_Block(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=kernel_size,padding=padding,stride=stride)
        self.TP1_inter = nn.MaxPool1d(kernel_size=2, stride=2)
        self.TP1_intra = nn.ModuleList(tp1_intra)
        
        self.TMC_block2 = TMC_Block(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=padding, stride=stride)
        self.TP2_inter = nn.MaxPool1d(kernel_size=2, stride=2)
        self.TP2_intra = nn.ModuleList(tp2_intra)
    
    def forward(self, x):
        """
            parameters :
                x         : list()
                    [face, Hands(left, right), Full-Frame, Pose]
                        face		: T X 256
                        both-hands	: T X 512
                        full-frame	: T X 512
                        pose 		: T X 256
            returns    : list()
                output     : list()
                    [inter_cue, intra_cue]
                        inter_cue   : T/4 x 1024
                        intra_cue   : T/4 x 1024
        """
        # convert size of each feature from (T x d) ==> (1 x d x T) for using nn.Conv1d and nn.MaxPool1d requiring 3 dimension input
        
        converted_x = list()
        for each_x in x:
            converted_x.append(each_x.unsqueeze(0).permute((0, 2, 1)).contiguous())

        inter_out1, intra_out1 = self.TMC_block1(torch.cat(converted_x, dim=1), converted_x)
        inter_out1 = self.TP1_inter(inter_out1)
        for i in range(len(intra_out1)):
            intra_out1[i] = self.TP1_intra[i](intra_out1[i])
        
        inter_out2, intra_out2 = self.TMC_block2(inter_out1, intra_out1)
        inter_out2 = self.TP2_inter(inter_out2)
        for i in range(len(intra_out2)):
            intra_out2[i] = self.TP2_intra[i](intra_out2[i])
        inter_out2 = inter_out2.squeeze(0).T.contiguous()
        
        converted_intra_out = list()
        
        for each_intra_out in intra_out2:
            converted_intra_out.append(each_intra_out.squeeze(0).T.contiguous())
        
        return inter_out2, intra_out2

#------------------------------------------------------------------------------------------------------------------------------------------------------
# TMC block

class TMC_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=1):
        """
            parameters:
                in_channels : list()
                    list of channels of each feature cues
                out_channels: list()
                    list of channels of each output from each feature cues
        """
        super(TMC_Block, self).__init__()
        
        self.inter_cue_path = Inter_Cue_path(in_channel=sum(in_channels),out_channel=sum(out_channels),
                                             kernel_size=kernel_size,padding=padding,stride=stride)
        
        self.intra_cue_path = Intra_Cue_path(in_channels=in_channels,out_channels=out_channels,
                                             kernel_size=kernel_size,padding=padding,stride=stride)
        
    def forward(self, x_inter_cue, x_intra_cue):
        """
            parameters:
                x_inter_cue : torch.Tensor size of (1 x sum(in_channels) x T)
                    input for inter cue path
                x_intra_cue : torch.Tensor size of (1 x sum(in_channels) x T)
                    input for intra cue path
            returns   :
                output_inter_cue    : torch.Tensor size of (1 x sum(out_channels) x T)
                    output for inter cue path
                output_intra_cue    : torch.Tensor size of (1 x sum(out_channels) x T)
                    output for intra cue path
        """
        
        output_intra_cue, inter_cue_input = self.intra_cue_path(x_intra_cue)
        
        output_inter_cue = self.inter_cue_path(x_inter_cue, inter_cue_input)
        
        return output_inter_cue, output_intra_cue
    

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Paths in TMC block

class Inter_Cue_path(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size=5, padding=2, stride=1):
        super(Inter_Cue_path, self).__init__()
        
        self.temporal_conv = nn.Conv1d(in_channel, out_channel//2,
                                       kernel_size=kernel_size, padding=padding, stride=stride)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, intra_cue):
        """
            parameters:
                x            : torch.Tensor size of (1 x in_channel x T)
                    input
                intra_cue    : torch.Tensor size of (1 x out_channel/2 x T)
                    intra cue injected by Intra Cue path
                    
            return    :
                output       : torch.Tensor size of (1 x out_channel x T)
                    output
        """
        x = self.temporal_conv(x)
        
        output = torch.cat([x, intra_cue], dim=1) # output = 1 x out_channel x T
        
        output = self.relu(output)
        
        return output
    
class Intra_Cue_path(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=1):
        """
            parameters:
                in_channels : list()
                    list of channels of each feature cues
                out_channels: list()
                    list of channels of each output from each feature cues
        """
        super(Intra_Cue_path, self).__init__()
        
        self.face        = nn.Sequential(nn.Conv1d(in_channels[0], out_channels[0], kernel_size=kernel_size,
                                                   padding=padding, stride=stride), nn.ReLU(inplace=True))
        self.both_hands  = nn.Sequential(nn.Conv1d(in_channels[1], out_channels[1], kernel_size=kernel_size,
                                                   padding=padding, stride=stride), nn.ReLU(inplace=True)) 
        self.full_frame  = nn.Sequential(nn.Conv1d(in_channels[2], out_channels[2], kernel_size=kernel_size,
                                                   padding=padding, stride=stride), nn.ReLU(inplace=True))
        self.pose        = nn.Sequential(nn.Conv1d(in_channels[3], out_channels[3], kernel_size=kernel_size,
                                                   padding=padding, stride=stride), nn.ReLU(inplace=True))
        
        self.TC_1 = nn.Conv1d(sum(out_channels), sum(out_channels)//2, kernel_size=1)
        
    def forward(self, x):
        """
            parameters :
                x : list()
                    list of features from each cues
                    [face, Hands(left, right), Full-Frame, Pose]
            returns    :
                output : list()
                    list of features from each cues after Temporal Conv
                    [face, Hands(left, right), Full-Frame, Pose]
        """
        face       = self.face(x[0])
        both_hands = self.both_hands(x[1])
        full_frame = self.full_frame(x[2])
        pose       = self.pose(x[3])
        
        output = [face, both_hands, full_frame, pose]
        
        inter_cue_input = self.TC_1(torch.cat(output, dim=1)) # size of (1 x sum(out_channels) x T)
        
        return output, inter_cue_input
    