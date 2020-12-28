%% ?????????your_denoising?????????PCA?image denoising???
%% ????????????PSNR?SSIM?????
%% ???????????????????????PSNR,SSIM?

clc;
clear;
addpath('Code');

profile  =   'normal';   
v        =   20; %% ???????
oI       =   double( imread('Images\house.tif') );

seed   =  0;
randn( 'state', seed );
noise      =   randn(size( oI ));
noise      =   noise/sqrt(mean2(noise.^2));
nI         =   oI + v*noise;
K          =   0;    % The width of the excluded boundaries, set to 20 to get the results in our paper

[ d_im ]   =  your_denoising( nI, oI, v, profile, K );
imwrite( d_im/255,'lena_denosied.tif','tif' );


%% PSNR and SSIM of noisy image
psnr_1    =   csnr(oI,nI,K,K);
ssim_1    =   cal_ssim( oI, nI, K, K );

%% PSNR and SSIM of denosied image
psnr_2    =   csnr(oI,d_im,K,K);
ssim_2    =   cal_ssim( oI, d_im, K, K );

