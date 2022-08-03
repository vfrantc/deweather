clear all;
close all;

input_folders = {'../data/epdn/outdoor',
                 '../data/epdn/test1',
                 '../data/ffa/outdoor',
                 '../data/gca_dehaze/outdoor',
                 '../data/gca_dehaze/test1',
                 '../data/gca_derain/rain1400',
                 '../data/gca_derain/test1',
                 '../data/mprnet/rain1400',
                 '../data/mprnet/test1',
                 '../data/refinednet/outdoor',
                 '../data/refinednet/test1',
                 '../data/transweather/outdoor',
                 '../data/transweather/rain1400',
                 '../data/transweather/rain_drop_test',
                 '../data/transweather/snowtest100k_L',
                 '../data/transweather/test1',
                 '../data/desnownet/Snow100K-L',
                 '../data/mmaxim/outdoor',
                 '../data/mmaxim/rain_drop_test',
                 '../data/mmaxim/test1_fog',
                 '../data/mmaxim/rain1400'
                 '../data/tskf/outdoor',
                 '../data/tskf/rain1400',
                 '../data/tskf/Snow100K-L',
                 '../data/tskf/test1'};

ground_folders = {'../data/test/outdoor/gt',
                  '../data/test/test1/gt',
                  '../data/test/outdoor/gt',
                  '../data/test/outdoor/gt',
                  '../data/test/test1/gt',
                  '../data/test/rain1400/gt',
                  '../data/test/test1/gt',
                  '../data/test/rain1400/gt',
                  '../data/test/test1/gt',
                  '../data/test/outdoor/gt',
                  '../data/test/test1/gt',
                  '../data/test/outdoor/gt',
                  '../data/test/rain1400/gt',
                  '../data/test/rain_drop_test/gt',
                  '../data/test/Snow100K-L/gt',
                  '../data/test/test1/gt',
                  '../data/test/Snow100K-L/gt',
                  '../data/test/outdoor/gt',
                  '../data/test/rain_drop_test/gt',
                  '../data/test/test1/gt',
                  '../data/test/rain1400/gt',
                  '../data/test/outdoor/gt',
                  '../data/test/rain1400/gt',
                  '../data/test/Snow100K-L/gt',
                  '../data/test/test1/gt'};

for i = 1:length(input_folders)
    disp('----------------------');
    disp(input_folders{i});
    disp(ground_folders{i});

    files = dir(input_folders{i});
    cnt = 0;
    psnr_ = 0.0;
    ssim_ = 0.0;
    for k = 1:length(files)
        file = files(k);
        if file.isdir
            continue
        end
            
        filename = [file.folder '/' file.name];


        gt_filename = [ground_folders{i} '/' file.name];
        if ~isfile(gt_filename)
            gt_filename = [ground_folders{i} '/' file.name(1:end-4) '.jpg'];
        end
        if ~isfile(gt_filename)
            gt_filename = [ground_folders{i} '/' file.name(1:end-4) '.png'];
        end

        image_gt = imread(gt_filename);
        [h, w, c] = size(image_gt);
        image = imread(filename);
        [ih, iw, ic] = size(image);
        if (ih ~= h) || (iw ~= w) 
            image = imresize(image, [h, w]);
        end
            
        %% COMPUTE SSIM
        ssim = compute_ssim(image, image_gt);
        ssim_ = ssim_ + ssim;

        %% COMPUTE PSNR
        peaksnr = compute_psnr(image, image_gt);
        psnr_ = psnr_ + peaksnr;

        cnt = cnt + 1;
    end
    disp(['psnr: ' string(psnr_/cnt)]);
    disp(['ssim:' string(ssim_/cnt)]);
end

function ssim_mean=compute_ssim(img1,img2)
    if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
    end

    if size(img2, 3) == 3
        img2 = rgb2ycbcr(img2);
        img2 = img2(:, :, 1);
    end
    ssim_mean = SSIM_index(img1, img2);
end

function psnr=compute_psnr(img1,img2)
    if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
    end

    if size(img2, 3) == 3
        img2 = rgb2ycbcr(img2);
        img2 = img2(:, :, 1);
    end

    imdff = double(img1) - double(img2);
    imdff = imdff(:);
    rmse = sqrt(mean(imdff.^2));
    psnr = 20*log10(255/rmse);
    
end


function [mssim, ssim_map] = SSIM_index(img1, img2, K, window, L)

%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================


if (nargin < 2 || nargin > 5)
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) || (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);	%
   K(1) = 0.01;								      % default settings
   K(2) = 0.03;								      %
   L = 255;                                  %
end

if (nargin == 3)
   if ((M < 11) || (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
	denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end