
%% the function calculate the luminance, color and texture saliency map for 2D images
%% new_i_smap is luminance saliency map; new_rg_smap and new_by_smap are color saliency maps; new_ac_smap is texture saliency map.
%% the code is based on the following paper:
% Yuming Fang, Zhenzhong Chen, Weisi Lin, and Chia-Wen Lin, 'Saliency Detection in the Compressed Domain for Adaptive Image Retargeting', IEEE Transactions on Image Processing (T-IP), 21(9): 3888-3901, 2012.
% or Yuming Fang, Zhenzhong Chen, Weisi Lin, and Chia-Wen Lin, 'Saliency-based Image Retargeting in the Compressed Domain', ACM International Conference on Multimedia 2011 (ACM MM11). 


function [new_i_smap, new_rg_smap, new_by_smap, new_ac_smap] = Image_Saliency(img)


%% resize the image into the new size with 16x*16y
r_img = img(:, :, 1);
g_img = img(:, :, 2);
b_img = img(:, :, 3);
[row, col] = size(r_img);
new_row = ceil(row/16) * 16;
new_col = ceil(col/16) * 16;
new_r_img = imresize(r_img, [new_row new_col], 'bilinear');
new_g_img = imresize(g_img, [new_row new_col], 'bilinear');
new_b_img = imresize(b_img, [new_row new_col], 'bilinear');
new_img(:, :, 1) = new_r_img;
new_img(:, :, 2) = new_g_img;
new_img(:, :, 3) = new_b_img;

ycc_img = rgb2ycbcr(new_img);

%% the image is transfered into YCbCr with 4:2:0
y_img = ycc_img(:, :, 1);
cb_img = imresize(ycc_img(:, :, 2), 0.5, 'bilinear');
cr_img = imresize(ycc_img(:, :, 3), 0.5, 'bilinear');


%% divide the image into 8*8 block
[y_row y_col] = size(y_img);
y_row_blk_num = y_row/8;
y_col_blk_num = y_col/8;
y_dct = zeros(y_row, y_col);

%% calculate the dct coefficients for Y channel.
%% obtain the YCrCb DCT DC coefficients which are used to calculate the saliency value for patches (8*8).
ycc_dc_coeff = zeros(y_row_blk_num, y_col_blk_num, 3);
y_dct_coeff = zeros(y_row_blk_num, y_col_blk_num, 8, 8);
for i = 1:y_row_blk_num
    for j = 1:y_col_blk_num
        y_dct((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8) = dct2(y_img((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8));
        ycc_dc_coeff(i, j, 1) = y_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        y_dct_coeff(i, j, :, :) = y_dct((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8);        
    end
end

%% calculate the dct coefficients for Cr and Cb channels.
[c_row c_col] = size(cb_img);
c_row_blk_num = c_row/8;
c_col_blk_num = c_col/8;
cb_dct = zeros(c_row, c_col);
cr_dct = zeros(c_row, c_col);
for i = 1:c_row_blk_num
    for j = 1:c_col_blk_num
        cb_dct((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8) = dct2(cb_img((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8));
        cr_dct((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8) = dct2(cr_img((i - 1) * 8 + 1 : i * 8, (j - 1) * 8 + 1 : j * 8));
        ycc_dc_coeff((i - 1) * 2 + 1, (j - 1) * 2 + 1, 2) = cb_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        ycc_dc_coeff((i - 1) * 2 + 1, (j - 1) * 2 + 2, 2) = cb_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        ycc_dc_coeff((i - 1) * 2 + 2, (j - 1) * 2 + 1, 2) = cb_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        ycc_dc_coeff((i - 1) * 2 + 2, (j - 1) * 2 + 2, 2) = cb_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        
        ycc_dc_coeff((i - 1) * 2 + 1, (j - 1) * 2 + 1, 3) = cr_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        ycc_dc_coeff((i - 1) * 2 + 1, (j - 1) * 2 + 2, 3) = cr_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        ycc_dc_coeff((i - 1) * 2 + 2, (j - 1) * 2 + 1, 3) = cr_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);
        ycc_dc_coeff((i - 1) * 2 + 2, (j - 1) * 2 + 2, 3) = cr_dct((i - 1) * 8 + 1, (j - 1) * 8 + 1);      
        
    end
end

y_ac_coeff = zeros(y_row_blk_num, y_col_blk_num);
for i = 1:y_row_blk_num
    for j = 1:y_col_blk_num
        y_ac_coeff(i, j) = sum(sum(y_dct_coeff(i, j, :, :))) - y_dct_coeff(i, j, 1, 1);
    end
end

rg_channel = ycc_dc_coeff(:, :, 2); %cr component
by_channel = ycc_dc_coeff(:, :, 3); % cb component
i_channel = ycc_dc_coeff(:, :, 1); % y component

%% the following variables are defined for calculating the distances between image patches
array_x = zeros(y_row_blk_num, y_col_blk_num);
array_y = zeros(y_row_blk_num, y_col_blk_num);
for i = 1:y_row_blk_num
    for j = 1:y_col_blk_num
        array_x(i, j) = i;
        array_y(i, j) = j;
    end
end

% compute the distances and center-surround differences between image patches
dist = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
csf = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
rg_diff = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
by_diff = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
i_diff = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
y_ac_diff = zeros(y_row_blk_num, y_col_blk_num, y_row_blk_num, y_col_blk_num);
for i = 1 : y_row_blk_num
    for j = 1 : y_col_blk_num
        dist(i, j, :, :) = sqrt((i - array_x).^2 + (j - array_y).^2);

        %% the parameter of Gaussian kernel is set as 20. It can be set as other values based on the image sizes
        csf(i, j, :, :) = (1/(20*(sqrt(2*pi))))*exp(-(dist(i, j, :, :).^2/(2*20.^2)));


%% center-surround difference calculation: d = |a-b|/|a+b|,
%         rg_diff(i, j, :, :) = abs(double(rg_channel(i, j)) - double(rg_channel))./ abs(double(rg_channel(i, j)) + double(rg_channel));
%         by_diff(i, j, :, :) = abs(double(by_channel(i, j)) - double(by_channel))./ abs(double(by_channel(i, j)) + double(by_channel));
%         i_diff(i, j, :, :) = abs(double(i_channel(i, j)) - double(i_channel)) ./ abs(double(i_channel(i, j)) + double(i_channel));
%         y_ac_diff(i, j, :, :) = abs(double(y_ac_coeff(i, j)) - double(y_ac_coeff)) ./ abs(double(y_ac_coeff(i, j)) + double(y_ac_coeff));
                
        
        %% center-surround difference calculation: d = |a-b|        
        rg_diff(i, j, :, :) = abs(double(rg_channel(i, j)) - double(rg_channel));
        by_diff(i, j, :, :) = abs(double(by_channel(i, j)) - double(by_channel));
        i_diff(i, j, :, :) = abs(double(i_channel(i, j)) - double(i_channel));
        y_ac_diff(i, j, :, :) = abs(double(y_ac_coeff(i, j)) - double(y_ac_coeff));             

        
    end
end

min_csf = min(min(min(min(csf))));
max_csf = max(max(max(max(csf))));
csf = (csf - min_csf)/(max_csf-min_csf);

rg_smap = zeros(y_row_blk_num, y_col_blk_num);
by_smap = zeros(y_row_blk_num, y_col_blk_num);
i_smap = zeros(y_row_blk_num, y_col_blk_num);
ac_smap = zeros(y_row_blk_num, y_col_blk_num);

%% saliency estimation by weighted center-surround differences
for i = 1 : y_row_blk_num
    for j = 1 : y_col_blk_num
        rg_smap(i, j) = sum(sum(rg_diff(i, j, :, :) .* csf(i, j, :, :)));
        by_smap(i, j) = sum(sum(by_diff(i, j, :, :) .* csf(i, j, :, :)));
        i_smap(i, j) = sum(sum(i_diff(i, j, :, :) .* csf(i, j, :, :)));
        ac_smap(i, j) = sum(sum(y_ac_diff(i, j, :, :) .* csf(i, j, :, :)));
    end
end

min_rg = min(min(rg_smap));
max_rg = max(max(rg_smap));
min_by = min(min(by_smap));
max_by = max(max(by_smap));
min_i = min(min(i_smap));
max_i = max(max(i_smap));
min_ac = min(min(ac_smap));
max_ac = max(max(ac_smap));

%% FOR NEW TEST
new_rg_smap = (rg_smap - min_rg)/(max_rg - min_rg);
new_by_smap = (by_smap - min_by)/(max_by - min_by);
new_i_smap = (i_smap - min_i)/(max_i - min_i);
new_ac_smap = (ac_smap - min_ac)/(max_ac - min_ac);

new_rg_smap = imresize(new_rg_smap, [row, col]);
new_by_smap = imresize(new_by_smap, [row, col]);
new_i_smap = imresize(new_i_smap, [row, col]);
new_ac_smap = imresize(new_ac_smap, [row, col]);

new_rg_smap = mat2gray(new_rg_smap);
new_by_smap = mat2gray(new_by_smap);
new_i_smap = mat2gray(new_i_smap);
new_ac_smap = mat2gray(new_ac_smap);

 clear new_img; 
 clear ycc_dc_coeff;
 clear y_ac_coeff;
 clear y_dct_coeff;

end






