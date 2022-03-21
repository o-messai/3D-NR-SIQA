function [im_crop_gray] = all_rgb2gray(img_crop_rgb)

for i=1:size(img_crop_rgb,4)
    
    im_crop_gray(:,:,:,i) = rgb2gray(img_crop_rgb(:,:,:,i));


end

