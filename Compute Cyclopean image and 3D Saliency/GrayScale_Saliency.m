
%% The code is used for extract the depth saliency for 3D saliency. Please note this file can be also used to extract saliency for gray-scale images.

%% input:grayscale image; output: saliency map

function final_smap = GrayScale_Saliency(img)


%% resize the image into the new size with 16x*16y for DCT coefficient extraction

[row, col] = size(img);
new_row = ceil(row/16) * 16;
new_col = ceil(col/16) * 16;
y_img = imresize(img, [new_row new_col], 'bilinear');

%% divide the image into 8*8 block
[y_row y_col] = size(y_img);
y_row_blk_num = y_row/8;
y_col_blk_num = y_col/8;
y_dct = zeros(y_row, y_col);

%% calculate the dct coefficients.
%% obtain the DCT DC coefficients which are used to calculate the saliency value for patches (8*8).
ycc_dc_coeff = zeros(y_row_blk_num, y_col_blk_num);
y_dct_coeff = zeros(y_row_blk_num, y_col_blk_num, 8, 8);
for i = 1:y_row_blk_num
    for j = 1:y_col_blk_num
        y_dct((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8) = dct2(y_img((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8));
        ycc_dc_coeff(i, j) = y_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        y_dct_coeff(i, j, :, :) = y_dct((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8);        
    end
end

y_channel = ycc_dc_coeff;

%% calculate the distances between image patches to be used to weight the certer-surround differences
array_x = zeros(y_row_blk_num, y_col_blk_num);
array_y = zeros(y_row_blk_num, y_col_blk_num);
for i = 1:y_row_blk_num
    for j = 1:y_col_blk_num
        array_x(i, j) = i;
        array_y(i, j) = j;
    end
end

% compute the Gaussian and differences between patches
dist = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
csf = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
rg_diff = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
for i = 1 : y_row_blk_num
    for j = 1 : y_col_blk_num
        dist(i, j, :, :) = sqrt((i - array_x).^2 + (j - array_y).^2);
        
        %% the parameter of Gaussian kernel is set as 20. csf is modeled by Gaussian function and used to weight the center-surround difference
        csf(i, j, :, :) = (1/(20*(sqrt(2*pi))))*exp(-(dist(i, j, :, :).^2/(2*20.^2)));             
        rg_diff(i, j, :, :) = abs(double(y_channel(i, j)) - double(y_channel))./(abs(double(y_channel(i, j)) + double(y_channel)));
    end
end


min_csf = min(min(min(min(csf))));
max_csf = max(max(max(max(csf))));
csf = (csf - min_csf)/(max_csf-min_csf);

y_smap = zeros(y_row_blk_num, y_col_blk_num);
for i = 1 : y_row_blk_num
    for j = 1 : y_col_blk_num
        y_smap(i, j) = sum(sum(rg_diff(i, j, :, :) .* csf(i, j, :, :)));
    end
end

min_y = min(min(y_smap));
max_y = max(max(y_smap));

%% normailization
final_smap = (y_smap - min_y)/(max_y - min_y);

clear new_img; 
clear ycc_dc_coeff;
clear y_dct_coeff;
end
