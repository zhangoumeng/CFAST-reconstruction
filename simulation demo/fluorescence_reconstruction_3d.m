function [g,cost] = fluorescence_reconstruction_3d(img,PSF,stopThreshold,maxIter,vis)

%% preprocessing image and PSF
rPSF = (size(PSF,1)-1)/2;

%%% avoid edge artifacts, uncomment for larger FOV
% img([1:rPSF,end-rPSF+1:end],:,:) = 0;
% img(:,[1:rPSF,end-rPSF+1:end],:) = 0;
%%%%

imgc = double([img(:,:,1),img(:,:,4);img(:,:,2),img(:,:,3)]);

for i = 1:size(PSF,4)
    PSFtmp(:,:,i) = [PSF(:,:,1,i),zeros(size(PSF,1),size(img,1)-size(PSF,1)),PSF(:,:,4,i);...
        zeros(size(img,1)-size(PSF,1),size(img,1)+size(PSF,1));...
        PSF(:,:,2,i),zeros(size(PSF,1),size(img,1)-size(PSF,1)),PSF(:,:,3,i)];
    PSFptmp(:,:,i) = rot90(PSFtmp(:,:,i),2);
end
PSF = PSFtmp;
PSFp = PSFptmp;

%% initialize
for i = 1:size(PSF,3)
    g(:,:,i) = double(mean(img,3))/size(PSF,3);
end

%% deconvolution
for iter = 1:maxIter
    % tic
    g = HT(imgc,PSFp,rPSF)./HTH(g,PSFp,PSF,rPSF).*g;
    diff = (H(g,PSF,rPSF) - imgc);
    cost(iter) = sum(diff(:).^2);
    if iter>1 && abs(cost(iter)-cost(iter-1))<cost(iter)*stopThreshold
        break;
    end
    
    if strcmp(vis,'on')

        figure(101); set(gcf,'position',[100,100,800,600]);
        subplot(2,2,1);
        imagesc(g(:,:,3)); 
        colorbar; axis image;
    
        subplot(2,2,2);
        imagesc(g(:,:,round((end+1)/2)));
        colorbar; axis image;
    
        subplot(2,2,3);
        imagesc(g(:,:,end-2));
        colorbar; axis image;
    
        subplot(2,2,4);
        plot(cost);
        title(['iter = ',num2str(iter)]);
    end
    % toc
end

end

%% functions for deconvolution
function out = HT(o,PSFp,rPSF)
r = size(o,1)/2;
out = nan(r,r,size(PSFp,3));
for i = 1:size(PSFp,3)
    outTmp = conv2(PSFp(:,:,i),o);
    out(:,:,i) = outTmp(r+rPSF+(1:r),r+rPSF+(1:r));
end
end

%
function out = H(g,PSF,rPSF)
r = size(g,1);
out = 0;
for i = 1:size(PSF,3)
    oTmp = conv2(PSF(:,:,i),g(:,:,i));
    out = out + oTmp(rPSF+(1:r*2),rPSF+(1:r*2));
end
end

%
function out = HTH(g,PSFp,PSF,rPSF)

r = size(g,1);
o = 0;
for i = 1:size(PSFp,3)
    oTmp = conv2(PSF(:,:,i),g(:,:,i));
    o = o + oTmp(rPSF+(1:r*2),rPSF+(1:r*2));
end

out = nan(r,r,size(PSFp,3));
for i = 1:size(PSFp,3)
    outTmp = conv2(PSFp(:,:,i),o);
    out(:,:,i) = outTmp(r+rPSF+(1:r),r+rPSF+(1:r));
end

end
