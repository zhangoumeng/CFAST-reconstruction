function PSF = generatePSF_z(zList)

r = 250;
k = 1;

for z = zList
    [Itmp,N] = computePSF_isotropic(z);
    I(:,:,:,k) = imresize(Itmp(N/2+(-r:r),N/2+(-r:r),:),.2);
    k = k + 1;
end
PSF = I/max(I(:));

end

%%
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