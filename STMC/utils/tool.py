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
        

        return torch.cat([row_coord, col_coord], dim=-1)
    
    
class cropping_patch(nn.Module):
    def __init__(self):
        super(cropping_patch, self).__init__()
        self.zero_padd_size = 24
        self.zero_padding = nn.ZeroPad2d(self.zero_padd_size)
        
    def forward(self, src, coord):
        """
        src    : T x C x H x W
        coord  : T x 14 (7 + 7)
        return:
            [0] : nose(Face)       16X16
            [1] : R-Wrist(R-Hand)  24X24
            [2] : L-Wrist(L-Hand)  24X24
        """
        H, W = src.size()[-2:]
        H -= 1
        W -= 1
        
        face_src, R_src, L_src = [], [], []
        row, col = coord[:, :7], coord[:, 7:]
        row = (row*H).int()
        row += self.zero_padd_size
        col = (col*W).int()
        col += self.zero_padd_size
        src = self.zero_padding(src)
        for each_src, each_row, each_col in zip(src, row, col):
            face = each_src[:, each_row[0] - 8 : each_row[0] + 8, each_col[0] - 8: each_col[0] + 8]
            R = each_src[:, each_row[1] - 12 : each_row[1] + 12, each_col[1] - 12: each_col[1] + 12]
            L = each_src[:, each_row[2] - 12 : each_row[2] + 12, each_col[2] - 12: each_col[2] + 12]
            face_src += [face]
            R_src += [R]
            L_src += [L]
                
                
        face_src = torch.stack(face_src, dim = 0)
        R_src = torch.stack(R_src, dim = 0)
        L_src = torch.stack(L_src, dim = 0)
        return face_src, L_src, R_src
    

    
class Squeeze(nn.Module):
    
    def __init__(self):
        super(Squeeze, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)
    
    
        
        
        
        
def SMC_weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.reset_parameters()
        