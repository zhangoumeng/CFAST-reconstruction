clear; close all;

pathName = 'data folder\'; % analyze all images ending with '_brightfield.tif' in this folder

addpath(pathName);

k = 1;
f = dir(pathName);
for i = 1:length(f)
    fNameTmp = f(i).name;
    if length(fNameTmp) > 16
        if strcmp(fNameTmp(end-15:end),'_brightfield.tif')
            fNameList{k} = fNameTmp(1:end-16);
            k = k + 1;
        end
    end
end
% images are tif stacks with 4 frames, each corresponding to a pupil block orientation

for fNameIdx = 1:length(fNameList)

    clearvars -except fNameIdx fNameList;

    lambda = 488e-9;
    
    fName = fNameList{fNameIdx};
    correction = 20; % cylindrical aberration correction, pre-calibrated
    
    pxSize = 1.85/20;
    
    F = @(x) ifftshift(fft2(fftshift(x)));  % Fourier Transform
    iF = @(x) ifftshift(ifft2(fftshift(x))); % Inverse Fourier Transform
    
    for i = 1:4
        img = fliplr(imread([fName,'_brightfield.tif'],i)); % flip brightfield image to match fluorescence images
        I = double(img(:,501:3500)); % original images are 3000x4000, crop the center part
        imSize = size(I,1);
    
        [u,v] = meshgrid(linspace(-2.65,2.65,imSize));
        [p,r] = cart2pol(u,v);
    
        u1 = 0;
        v1 = 0;
        r1FT = zeros(imSize);
        switch i
            case 1
                v1 = -round(imSize/5);
                H = 1j*sign(v);
                H(imSize/2+1,:) = 0;
                r1FT(imSize/2+1-v1,imSize/2+1) = 1;
            case 2
                u1 = -round(imSize/5);
                H = 1j*sign(u);
                H(:,imSize/2+1) = 0;
                r1FT(imSize/2+1,imSize/2+1-u1) = 1;
            case 3
                v1 = round(imSize/5);
                H = -1j*sign(v);
                H(imSize/2+1,:) = 0;
                r1FT(imSize/2+1-v1,imSize/2+1) = 1;
            case 4
                u1 = round(imSize/5);
                H = -1j*sign(u);
                H(:,imSize/2+1) = 0;
                r1FT(imSize/2+1,imSize/2+1-u1) = 1;
        end
        r1 = iF(r1FT);

        x_re = .5*log(I);
        x_im = iF(F(x_re).*H);

        Stmp = F(exp(x_re+1j*x_im).*r1);
        S(:,:,i) = circshift(Stmp,[v1,u1]);
    
    end
    
    t = 1/180*pi; % slightly tilted pupil block, pre-calibrated 
    Sfull = zeros(imSize);
    Stmp = S(:,:,1);
    pTmp = angle(exp(1j*(p+pi)));
    idx = logical((pTmp>t).*(pTmp<pi/2+t));
    Sfull(idx) = Stmp(idx);

    Stmp = S(:,:,2);
    pTmp = angle(exp(1j*(p-pi/2)));
    idx = logical((pTmp>t).*(pTmp<pi/2+t));
    Sfull(idx) = Stmp(idx);

    Stmp = S(:,:,3); 
    pTmp = p;
    idx = logical((pTmp>t).*(pTmp<pi/2+t));
    Sfull(idx) = Stmp(idx);

    Stmp = S(:,:,4);
    pTmp = angle(exp(1j*(p+pi/2)));
    idx = logical((pTmp>t).*(pTmp<pi/2+t));
    Sfull(idx) = Stmp(idx);

    Sfull = circshift(Sfull,[0,1]);
    
    imagesc(abs(Sfull)); clim([0,.1]);

    k = 0;
    for zNum = -50:50 % generate digitally refocusing stack from -50 to 50 microns
        k = k + 1;
        z = zNum/1e6;
        SfullTmp = Sfull.*exp(1j*2*pi/lambda*z*sqrt(1-r.^2)).*...
            exp(1j*2*pi/lambda*correction*1e-6*sqrt(1-u.^2));
        SfullTmp(r>.42) = 0;
        SfullTmp = circshift(SfullTmp,[0,-1]);
        s = rot90(F(SfullTmp),2);
    
        img = abs(s)/max(abs(s(:)));
        img = uint8(round(img*255));
        if correction ~= 0
            fNameA = [fName,'_zStack_correction.tif'];
            fNameP = [fName,'_zStack_phase_correction.tif'];
        else
            fNameA = [fName,'_zStack.tif'];
            fNameP = [fName,'_zStack_phase.tif'];
        end
        if k == 1
            imwrite(img,fNameA);
        else
            imwrite(img,fNameA,'writemode','append');
        end
    
        img = angle(s)/pi/2+.5;
        img = uint8(round(img*255));
        if k == 1
            imwrite(img,fNameP);
        else
            imwrite(img,fNameP,'writemode','append');
        end
    end
end