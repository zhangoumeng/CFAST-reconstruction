# Network class

import torch
import torch.nn as nn

class DeconNet(nn.Module):
    def __init__(self, img, PSF, PSFR, rPSF, r):
        super(DeconNet, self).__init__()
        self.rPSF = rPSF
        self.r = r
        self.PSF_fft = torch.fft.fftn(PSF, dim=(-2, -1), s=(r,r))
        self.PSFR_fft = torch.fft.fftn(PSFR, dim=(-2, -1), s=(r,r))

        img_fft = torch.fft.fftn(img, dim=(-2, -1), s=(r,r)).expand(-1, 41, -1, -1)
        HT = torch.sum(torch.fft.ifftn(img_fft * self.PSFR_fft, dim=(-2, -1)) , dim=0, keepdim=True)
        self.HT_abs = abs(HT[:,:,rPSF:-rPSF,rPSF:-rPSF])

    def forward(self, imstack):
        imstack_fft = torch.fft.fftn(imstack, dim=(-2,-1), s=(self.r,self.r)).expand(4, -1, -1, -1)
        H = torch.sum(torch.fft.ifftn(imstack_fft * self.PSF_fft, dim=(-2,-1)), dim=1, keepdim=True) 
        H_fft = torch.fft.fftn(H, dim=(-2,-1)).expand(-1, 41, -1, -1)
        HTH = torch.sum(torch.fft.ifftn(H_fft * self.PSFR_fft, dim=(-2, -1)) , dim=0, keepdim=True)
        imstack = self.HT_abs / (abs(HTH[:,:,2*self.rPSF:,2*self.rPSF:])) * imstack # weird shift...
        
        return imstack