% =================================================================
%  University of Constantine-1
%  Automatic and Robotic Laboratory
%  Université de lumière Lyon 2, Laboratoire LIIRS
%  Copyright(c) 2022  MESSAI Oussama
%  e-mail: oussama.messai@univ-lyon2.fr
%  All Rights Reserved.
% -----------------------------------------------------------------
%% The code is used to compute Cyclopean image and to extract the corresponding  3D slaiency map. Please cite the following papers when using the code.
% O. Messai, A. Chetouani, F. Hachouf, and Z. Ahmed Seghir, “3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images”, in Neurocomputing Journal, January 06, 2022, Elsevier.

%clc;clear;close all;
addpath ( genpath ( 'Files mat' ) );
addpath ( genpath ( 'MJ3DQA' ) );                                          % load MJ3DQA function
load('IVCmosPh2.mat');
load Filename_W2                                                  % load database information and DMOS data
%h = waitbar(0,'Please wait...');
i = 453;
max_disp = 25;
index = 1;

%% Mesure the quality of the Stereo Image
for iPoint = 453:460
    %READ A DISTORTED IMAGE
              
            %Id = imread(['/home/temp/Bureau/Oussama/Database/WaterlooIVC3DPhaseII/WaterlooIVC3DPhaseII/3D/Session3DIQ/' Filename{iPoint}]);
            %sizeTemp = size(Id);
           % imDL = Id(:,1:sizeTemp(2)/2,:);
           % imDR = Id(:,sizeTemp(2)/2+1:end,:); 
            %figure, imshow(Id);
            %imDL = rgb2gray(imDL);
			%imDR = rgb2gray(imDR);
			
    % OBJECTIVE METRIC FUNCTION
    %[fdsp, dmap_test, confidence, diff] = mj_stereo_SSIM(imDL,imDR, max_disp);
    %[CI]  = MJ3DQA(imDL(:,:,1),imDR(:,:,1)); %RED
    %Cyclopean_W2(:,:,1,i) = CI;
    %Disparity_L1(:,:,1,i) = dmap_test;
    
    %[CI]  = MJ3DQA(imDL(:,:,2),imDR(:,:,2)); %GREEN
    %Cyclopean_W2(:,:,2,i) = CI;
    %Disparity_L1(:,:,2,i) = dmap_test;
    
    %[CI]  = MJ3DQA(imDL(:,:,3),imDR(:,:,3)); %BLUE
    %Cyclopean_W2(:,:,3,i) = CI;
    %Disparity_L1(:,:,3,i) = dmap_test;
    
    
   % [fdsp, dmap_test, confidence, diff] = mj_stereo_SSIM(rgb2gray(imDL),rgb2gray(imDR), max_disp);
   % k = find(dmap_test(:)==0); dmap_test(k)=1; % Take off zero values
    %Disparity_W2(:,:,i) = dmap_test;
    %D_add = max(dmap_test(:))- min(dmap_test(:));
    %dmap_test = dmap_test + D_add;
    
%     V = 294.64; I = 6.3; Rx=75.9; W=129.7;
%     [K, L] = size(dmap_test);
%     for m=1:K
%         for n=1:L
%            depth_map(m,n) = V ./ (1 + ((I*Rx)./(dmap_test(m,n).*W)));  
%         end
%     end
     depth_map= (300*6.5)./Disparity_W2(:,:,i);                                     % 
                                         %depth_map= (195.15*10)./dmap_test; depth_map = V / (1 + ((I*Rx)/(dmap_test*W))
                                         %The study was conducted using a Panasonic 58 in. 3D
                                         %TV with active shutter glasses. The viewing distance was
                                         %set at 116 in., which is four times the screen height.
    img = uint8(Cyclopean_W2(:,:,:,index));  % I = 6.3cm, V = 294.64 cm, Rx=75.9cm ,W=129.7
    depth = uint8(depth_map);
    
    %% If the image size is over 600*600, you can resize the input image into a smaller size such as 300*300 or 300*400 for low time cost.
    
    [row col dim] = size(img);
    img = imresize(img, 0.5);
    depth = imresize(depth, 0.5);
    
    %% depth saliency calculation
    %disp('calculate the depth saliency...');
    %figure, imshow(depth);
    depth_smap = GrayScale_Saliency(depth);
    
    %% Luminance, color and texture saliency calcualtion for 2D images.
    %disp('calculate the luminance, color and texture saliency...')
    [y_smap, cr_smap, cb_smap, texture_smap] = Image_Saliency(img);
    
    %% resize to the original size
    y_smap = imresize(y_smap, [row col]);
    cr_smap = imresize(cr_smap, [row col]);
    cb_smap = imresize(cb_smap, [row col]);
    texture_smap = imresize(texture_smap, [row col]);
    depth_smap = imresize(depth_smap, [row col]);
    
    y_smap = mat2gray(y_smap);
    cr_smap = mat2gray(cr_smap);
    cb_smap = mat2gray(cb_smap);
    texture_smap = mat2gray(texture_smap);
    depth_smap = mat2gray(depth_smap);
    
    %% Saliency map for 2D images
    %disp('2D saliency prediction....');
    %smap_2d = (y_smap + cr_smap + cb_smap + texture_smap)/4.0;
    %smap_2d = mat2gray(smap_2d);
    %figure, imagesc(smap_2d);
    %% saliency enhancement by human visual sensitivity
    %norm_smap_2d = norm_operation(smap_2d);
    
    %% saliency map for 3D images
    %disp('3D saliency prediction....')
    smap_3d = (y_smap + cr_smap + cb_smap + texture_smap + 1.5.*depth_smap)/5.0;
    smap_3d = mat2gray(smap_3d);
    %figure, imagesc(smap_3d);
    
      %I = imtile({smap_2d, smap_3d}); 
     %figure(1), subplot(3,1,1), imagesc(I), title('Saliency 2D vs. Saliency 3D')
     %figure(1), subplot(3,1,2), imshow(img), title(num2str(iPoint))
     %figure(1), subplot(3,1,3), imshow(depth), title('Estimated depth')
    %% center bias map, please note the parameter might influence the performance of saliency prediction
    center_bias_map = CenterBias_Model(depth_smap, 10);
    center_bias_map = mat2gray(center_bias_map);
    smap_3d = mat2gray(0.1*center_bias_map + 0.9*smap_3d);
    norm_smap_3d = norm_operation(smap_3d);
    
    %% saliency maps for different feature maps
%     imwrite(y_smap, './Smap_Results/y_smap.jpg');
%     imwrite(cr_smap, './Smap_Results/cr_smap.jpg');
%     imwrite(cb_smap, './Smap_Results/cb_smap.jpg');
%     imwrite(texture_smap, './Smap_Results/texture_smap.jpg');
%     imwrite(depth_smap, './Smap_Results/depth_smap.jpg');
%     
%     %% saliency maps for 2D/3D images
%     imwrite(norm_smap_2d, './Smap_Results/smap_2d.jpg');
%     imwrite(norm_smap_3d, './Smap_Results/smap_3d.jpg');

   %% Display Saliency 2D vs Saliency 3D
   
%    MAP_3D = uint8(norm_smap_3d);
%    MAP_2D = uint8(norm_smap_2d);
%    
%    MAP_3D = double(MAP_3D);
%    MAP_2D = double(MAP_2D);
%    
%    H1 = find(MAP_3D(:)==1); MAP_3D(H1)=1.5;
%    H0 = find(MAP_3D(:)==0); MAP_3D(H0)=0.5;
%    
%    M1 = find(MAP_2D(:)==1); MAP_2D(M1)=1.5; 
%    M0 = find(MAP_2D(:)==0); MAP_2D(M0)=0.5;
   
%    figure,
%    subplot(1,2,1), imshow(uint8(MAP_2D.*double(img))), title('Saliency 2D')
%    subplot(1,2,2), imshow(uint8(MAP_3D.*double(img))), title('Saliency 3D')
   
   %Saliency_2d_W2(:,:,i) = norm_smap_2d;
   Saliency_3d_W2(:,:,i) = norm_smap_3d;
   disp(iPoint);
    %waitbar(iPoint/460);
    i = i+1;
    index = index + 1;
end
%close(h);
