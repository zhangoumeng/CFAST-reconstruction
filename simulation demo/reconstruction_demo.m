clear; close all;

%% simulate a spherical object and simulate images
pxSize = 1.85/20;
rSphere = 15; % micron
r = 400;

[X,Y,Z] = meshgrid(pxSize*(-r:r),pxSize*(-r:r),0:0.5:20);
X(X==0) = eps;
Y(Y==0) = eps;
Z(Z==0) = eps;

obj = zeros(size(X));

for p = 0:45:165
    t = atan(X./(Y*sind(p)+Z*cosd(p)));
    x1 = rSphere*sin(t);
    y1 = rSphere*cos(t)*sind(p);
    z1 = rSphere*cos(t)*cosd(p);
    x2 = rSphere*sin(t+pi);
    y2 = rSphere*cos(t+pi)*sind(p);
    z2 = rSphere*cos(t+pi)*cosd(p);
    dist(:,:,:,1) = sqrt((X-x1).^2+(Y-y1).^2+(Z-z1).^2);
    dist(:,:,:,2) = sqrt((X-x2).^2+(Y-y2).^2+(Z-z2).^2);
    score = 1./(min(dist,[],4)+.05);
    obj = obj + score;
end

%% plot object
cMap = cool(size(obj,3));

I0 = 0;
for i = 1:size(obj,3)
    I0 = I0 + cat(3,obj(:,:,i)*cMap(i,1),obj(:,:,i)*cMap(i,2),obj(:,:,i)*cMap(i,3)); 
end
figure(1); set(gcf,'position',[100,100,800,300]); subplot(1,3,1);
imagesc(I0/max(I0(:))*5); % *5 to improve contrast
axis image; set(gca,'xtick',[],'ytick',[])
title('object (color = z)');

%% generate PSF
rPSF = 50;
k = 1;
for z = (0:0.5:20)*1e-6
    [Itmp,N] = computePSF_isotropic(z);
    I(:,:,:,k) = Itmp(N/2+(-rPSF:rPSF),N/2+(-rPSF:rPSF),:);
    k = k + 1;
end
PSF = I/max(I(:));

%% simulate (downsampled) images
imgTmp = H(obj,squeeze(PSF(:,:,1,:)),rPSF);
img(:,:,1) = imresize(imgTmp,.2);
imgTmp = H(obj,squeeze(PSF(:,:,2,:)),rPSF);
img(:,:,2) = imresize(imgTmp,.2);
imgTmp = H(obj,squeeze(PSF(:,:,3,:)),rPSF);
img(:,:,3) = imresize(imgTmp,.2);
imgTmp = H(obj,squeeze(PSF(:,:,4,:)),rPSF);
img(:,:,4) = imresize(imgTmp,.2);

figure(1); subplot(1,3,2);
imagesc([img(:,:,1),img(:,:,4);img(:,:,2),img(:,:,3)]);
axis image; set(gca,'xtick',[],'ytick',[])
title('images');

% downsample PSF for deconvolution
k = 1;
for i = 1:2:size(PSF,4)
    PSF_ds(:,:,:,k) = imresize(PSF(:,:,:,i),.2);
    k = k + 1;
end
PSF_ds = PSF_ds/sum(PSF_ds(:));

%% deconvolution
[g,cost] = fluorescence_reconstruction_3d(img,PSF_ds,1e-5,200,'on');

% visualization
g = g/max(g(:));

cMap = cool(size(g,3));

I0 = 0;
for i = 1:21
    I0 = I0 + cat(3,g(:,:,i)*cMap(i,1),g(:,:,i)*cMap(i,2),g(:,:,i)*cMap(i,3)); 
end
figure(1); subplot(1,3,3);
imagesc(I0/max(I0(:))*2); % *2 to improve contrast
axis image; set(gca,'xtick',[],'ytick',[])
title('reconstruction');




%% convolution
function out = H(g,PSF,rPSF)
r = size(g,1);
out = 0;
for i = 1:size(PSF,3)
    oTmp = conv2(PSF(:,:,i),g(:,:,i));
    out = out + oTmp(rPSF+(1:r),rPSF+(1:r));
end
end

%% vectorial PSF model
function [I,N] = computePSF_isotropic(z)

lambda = 525e-9;
NA = .42;
bfp_radius = 64;
M = 20;
N = round(lambda*M*bfp_radius/NA/(1850e-9));
if rem(N,2) == 1
    N = N+1;
end

dx = 1850e-9/M;

[eta,xi] = meshgrid(linspace(-1/(2*dx),1/(2*dx),N),linspace(-1/(2*dx),1/(2*dx),N));

xBFP = lambda*eta;
yBFP = lambda*xi;
[phi,rho] = cart2pol(xBFP,yBFP);
rho_max = NA; %pupil region of support determined by NA and imaging medium r.i.

k1 = (2*pi/lambda);
theta1 = asin(rho);

Esx = -sin(phi);
Esy = cos(phi);
Epx = cos(phi).*cos(theta1);
Epy = sin(phi).*cos(theta1);
Epz = -sin(theta1);

Exx = (1./sqrt(cos(theta1))).*(cos(phi).*Epx - sin(phi).*Esx).*exp(1i*k1*z*cos(theta1));
Exy = (1./sqrt(cos(theta1))).*(cos(phi).*Epy - sin(phi).*Esy).*exp(1i*k1*z*cos(theta1));
Exz = (1./sqrt(cos(theta1))).*(cos(phi).*Epz).*exp(1i*k1*z*cos(theta1));


Exx(rho >= rho_max) = 0;
Exy(rho >= rho_max) = 0;
Exz(rho >= rho_max) = 0;

mask1 = ones(N);
mask1(xBFP>0) = 0;
mask1(yBFP>0) = 0;
imgExx_1 = fftshift(fft2(Exx.*mask1));
imgExy_1 = fftshift(fft2(Exy.*mask1));
imgExz_1 = fftshift(fft2(Exz.*mask1));

mask2 = ones(N);
mask2(xBFP>0) = 0;
mask2(yBFP<0) = 0;
imgExx_2 = fftshift(fft2(Exx.*mask2));
imgExy_2 = fftshift(fft2(Exy.*mask2));
imgExz_2 = fftshift(fft2(Exz.*mask2));

mask3 = ones(N);
mask3(xBFP<0) = 0;
mask3(yBFP<0) = 0;
imgExx_3 = fftshift(fft2(Exx.*mask3));
imgExy_3 = fftshift(fft2(Exy.*mask3));
imgExz_3 = fftshift(fft2(Exz.*mask3));

mask4 = ones(N);
mask4(xBFP<0) = 0;
mask4(yBFP>0) = 0;
imgExx_4 = fftshift(fft2(Exx.*mask4));
imgExy_4 = fftshift(fft2(Exy.*mask4));
imgExz_4 = fftshift(fft2(Exz.*mask4));

I(:,:,1) = abs(imgExx_1).^2+abs(imgExy_1).^2+abs(imgExz_1).^2;
I(:,:,2) = abs(imgExx_2).^2+abs(imgExy_2).^2+abs(imgExz_2).^2;
I(:,:,3) = abs(imgExx_3).^2+abs(imgExy_3).^2+abs(imgExz_3).^2;
I(:,:,4) = abs(imgExx_4).^2+abs(imgExy_4).^2+abs(imgExz_4).^2;

end