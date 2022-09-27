import torch
import torch.nn as nn


class Soft_Argmax(nn.Module):
    def __init__(self):
        super(Soft_Argmax, self).__init__()

    def forward(self, x):
        m = nn.Softmax(dim=-1)
        """
        B : the number of batch size ; T
        C : the channel size
        H : height
        W : width

        x.size() : ( B, C, H, W )
        """
        B, C, H, W = x.size()

        # index value generating
        coord = torch.arange(H).to(x.device)
        temp = x.view(B, C, H*W)
        temp = m(temp)
        softed = temp.view(B, C, H, W)

        # summing softmax value by row and column index
        row_sum = torch.sum(softed, dim=-2)/(H-1)
        col_sum = torch.sum(softed, dim=-1)/(W-1)

        row_coord = torch.sum(row_sum * coord, dim=-1)
        col_coord = torch.sum(col_sum * coord, dim=-1)
        

        return (row_coord, col_coord, B)
    
    
class cropping_patch(nn.Module):
    def __init__(self):
        super(cropping_patch, self).__init__()

    def forward(self, src, row, col, Height, Width):
        """
        src : T, C, H, W
        row : T, 7
        [0] : nose(Face)       16X16
        [1] : R-Wrist(R-Hand)  24X24
        [2] : L-Wrist(L-Hand)  24X24
        """
        face_src, R_src, L_src = [], [], []
        zero = nn.ZeroPad2d(20)
        row = (row*Height).int()
        row += 20
        col = (col*Width).int()
        col += 20
        src = zero(src)
        for t, s in enumerate(src):
            face = s[:, row[t, 0] - 8 : row[t, 0] + 8, col[t, 0] - 8: col[t, 0] + 8]
            R = s[:, row[t, 1] - 12 : row[t, 1] + 12, col[t, 1] - 12: col[t, 1] + 12]
            L = s[:, row[t, 2] - 12 : row[t, 2] + 12, col[t, 2] - 12: col[t, 2] + 12]
            face_src += [face]
            R_src += [R]
            L_src += [L]
            if R.size() != torch.randn(256, 24, 24).size():
                print(t, 'R:',R.size())
                print(row[t, 1]," or ",col[t, 1])
            elif L.size() != torch.randn(256, 24, 24).size():
                print(t, 'L:',L.size())
                print(row[t, 2]," or ",col[t, 2])
            elif face.size() != torch.randn(256, 16, 16).size():
                print(t, 'Face:',face.size())
                print(row[t, 0]," or ",col[t, 0])
        face_src = torch.stack(face_src)
        R_src = torch.stack(R_src)
        L_src = torch.stack(L_src)
        return face_src, L_src, R_src
    

    
class Squeeze(nn.Module):
    
    def __init__(self):
        super(Squeeze, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)
    
    
class TMC_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, cue_num, cut_point):
        super(TMC_Block, self).__init__()
        
        self.make_block(in_channels, out_channels)
        self.in_channels = in_channels
        self.Cn = cue_num
        
    def forward(self, prev_inter_cue, prev_intra_cue):
        
        intra_cue_path = self.block[0]
        TC_1_layer = self.block[1]
        inter_cue_path = self.block[2]
        
        intra_C = []
        inter_C = []
        
        """
        Intra-Cue Path
        
        prev_intra_cue : (T X sum(in_channels))
        intra_C : (T X out_channels)
        """
        
        start_index = 0
        for i in range(self.Cn):
            x = prev_inter_cue[:, start_index:start_index + self.in_channels[i] ]	# cutting intra concatenated cue features
            intra_C.append(intra_cue_path[i](x))
            start_index = self.in_channels[i]
        
        intra_C = torch.cat(intra_C, dim=1)
        intra_C = intra_cue_path[self.Cn](intra_C)
        
        """
        Inter-Cue Path
        
        prev_inter_cue : (T X sum(in_channels))
        intra_f : (T X out_channels)
        """
        
        TC_1_output = TC_1_layer(intra_C)
        
        inter_C = torch.cat([inter_cue_path[0](prev_inter_cue), TC_1_output], dim=1)
        
        inter_C = inter_cue_path[1](inter_C)
        
        return inter_C, intra_C
        
        
    def make_block(self, in_channels, out_channels):
        
        intra_cue_path = []
        inter_cue_path = []
        
        for i in range(self.Cn):
            intra_cue_path.append(nn.Conv1d(in_channels[i], int(out_channels/self.Cn), kernel_size = 5, padding = 2))
        intra_cue_path.append(nn.ReLU(inplace=True))
        intra_cue_path = nn.ModuleList(intra_cue_path)
        
        TC_1_layer = nn.Conv1d(sum(in_channels), int(out_channels/2), kernel_size=1)
        
        inter_cue_path = nn.ModuleList([nn.Conv1d(sum(in_channels), int(out_channels/2), kernel_size = 5, padding = 2), nn.ReLU(inplace=True)])
        
        self.block = nn.ModuleList([intra_cue_path, TC_1_layer, inter_cue_path])
        
        
        
        
        
        
        
def SMC_weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.reset_parameters()
        