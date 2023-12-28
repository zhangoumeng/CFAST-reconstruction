import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from skimage import io
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.io import savemat

from network import DeconNet

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == "__main__":
    # In[]  Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--model_opt", default="None", type=str) # "jit". "compile", "None"
    parser.add_argument("--use_amp", default=True, type=bool)
    parser.add_argument("--imsize", default=620, type=int)

    args = parser.parse_args()

    fPath = './data'
    PSFPath = './utils'
    fName = 'data_fluorescence.tif'
    # .tif stack with 4 slices, each corresponds to a pupil block orientation
    PSFName = 'PSF.mat'
    
    num_epochs = args.num_epochs
    is_opt = args.model_opt
    use_amp = args.use_amp
    imsize = args.imsize
    
    mat = sio.loadmat(os.path.join(PSFPath,PSFName))
    PSF = torch.tensor(mat['PSF']).to(torch.float32)
    PSFR = torch.tensor(mat['PSFR']).to(torch.float32)
    PSF = PSF.permute(2,3,0,1)
    PSFR = PSFR.permute(2,3,0,1) # this is PSF rotated by 180 degrees
    
    PSFsize = PSF.shape[2]

    # In[]  Load image
    img = torch.from_numpy(np.zeros((4, 1, 600, 600))).cuda()
    for i in range(4):
        imgTmp = io.imread(os.path.join(fPath,fName), img_num=i)
        imgTmp = np.double(imgTmp).astype(np.float32)
        img[i,0,:,:] = torch.from_numpy(imgTmp)
        
    del imgTmp
        
    rPSF = int((PSF.shape[2]-1)/2)
    r = imsize

    # In[]  initialization
    g = (img[0, 0,:, :] / PSF.shape[1]).repeat(PSF.shape[1], 1, 1).to(torch.float32).unsqueeze(0)
        
    model = DeconNet(img, PSF, PSFR, rPSF, r)
        
    if is_opt == "jit":
        model_fn = torch.jit.trace(model, g)
    elif is_opt == "compile":
        model_fn = torch.compile(model, backend="inductor")
    else:
        model_fn = model
    
    # In[] Iterative Deconvolution
    for epoch in tqdm(range(num_epochs)):
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            g = model_fn(g)
        torch.cuda.synchronize()

    # In[] save to mat file
    g_numpy = g.cpu().numpy()
    savemat('result.mat', {'g': g_numpy})

    # In[] plot results
    plt.figure(figsize=(6, 4), dpi=300)
    plt.subplot(1,3,1)
    plt.imshow(g_numpy[0,10,:,:], cmap='gray')
    plt.axis('image')
    plt.axis('off')
    plt.title('-10 micron')
    plt.clim(0, 4e3) 

    plt.subplot(1,3,2)
    plt.imshow(g_numpy[0,20,:,:], cmap='gray')
    plt.axis('image')
    plt.axis('off')
    plt.title('10 micron')
    plt.clim(0, 4e3) 

    plt.subplot(1,3,3)
    plt.imshow(g_numpy[0,30,:,:], cmap='gray')
    plt.axis('image')
    plt.axis('off')
    plt.title('10 micron')
    plt.clim(0, 4e3) 
    
    plt.show()