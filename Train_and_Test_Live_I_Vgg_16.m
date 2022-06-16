% =================================================================
%  University of Constantine-1
%  Automatic and Robotic Laboratory
%  Université de lumière Lyon 2, Laboratoire LIIRS
%  Copyright(c) 2022  MESSAI Oussama
%  e-mail: oussama.messai@univ-lyon2.fr
%  All Rights Reserved.
% -----------------------------------------------------------------
%% The code is used to train and test the deep vgg16 model. Please cite the following papers when using the code.
% O. Messai, A. Chetouani, F. Hachouf, and Z. Ahmed Seghir, “3D Saliency guided Deep Quality predictor for No-Reference Stereoscopic Images”, in Neurocomputing Journal, January 06, 2022, Elsevier.

clc;clear;close all;
addpath ( genpath ( 'mat files' ) ); 
addpath ( 'Train codes for other databases' );
load('data.mat');
load('norm_dL1.mat') % Normalized DMOS
load Cyclopean_L1.mat % Load the Cyclyopean image
load Saliency_3d_L1.mat % Loead the corresponding 3D Saliency
warning('off')

index = 1; X = []; Y=[]; %Percentage = 100;
for Sal_Value = [0.3:0.1:0.3] % Saliency threshold fixed to 0.3
    %[0.1:0.1:0.9]
    Dmos_new = []; Stock_img = {};
    %% Extract the Saliency from the Cyclopean Image
    parfor iPoint=1:365
        
        [ img_cropped, S ] = Crop_Saliency_Percentage(Cyclopean_L1(:,:,:,iPoint),Saliency_3d_L1(:,:,iPoint),31,31,Sal_Value,80);
        %[ img_cropped, S ] = Crop_Saliency(Cyclopean_L2(:,:,:,iPoint),Saliency_3d_L2(:,:,iPoint),31,31,Sal_Value);
        Stock_img{iPoint}  = [img_cropped(:,:,:,:)];
        
        OUT_Score = repmat(norm_dL1(iPoint),S,1); % prepare norm. dmos for training
        %T_output = [T_output;OUT_Score];
        Stock_normdL1{iPoint} = [OUT_Score];
        
        D_Score = repmat(dmos(iPoint),S,1); % prepare Dmos for correlation.
        Dmos_new = [Dmos_new; D_Score];
        Stock_Dmos_new{iPoint} = [D_Score];
        %T_input = cat(4,T_input, Stock_img{iPoint}); % For training input
    end
    disp(size(Dmos_new)); disp(Sal_Value);
    repeat = 2; m = 1;  k = 2; % number of folds
    for R=1:repeat
        %% Divide the Dataset as the five type distortions inputs
        [JPEG_crop, JP2K_crop, WN_crop, BLUR_crop, FF_crop]      = Divide_distortions_L1(Stock_img);
        [JPEG_dmos, JP2K_dmos, WN_dmos, BLUR_dmos, FF_dmos]      = Divide_distortions_L1(Stock_Dmos_new);
        [JPEG_normd, JP2K_normd, WN_normd, BLUR_normd, FF_normd] = Divide_distortions_L1(Stock_normdL1);
        
        % clear Stock_img Stock_Dmos_new Stock_normdL2; % Clear for memory
        
        
        Stock_img_new = [WN_crop, JP2K_crop, JPEG_crop, BLUR_crop, FF_crop]; % stock number of extracted patches from each image
        
        for j=1:365
            Patches_nbr(j) = size(Stock_img_new{j},4);
        end
        
        clear Stock_img_new; % Clear for memory
        
        [JPEG_crop_input] = concatenate_data(JPEG_crop,4);  clear JPEG_crop;
        [JP2K_crop_input] = concatenate_data(JP2K_crop,4);  clear JP2K_crop;
        [WN_crop_input] = concatenate_data(WN_crop,4);      clear WN_crop;
        [BLUR_crop_input] = concatenate_data(BLUR_crop,4);  clear BLUR_crop;
        [FF_crop_input] = concatenate_data(FF_crop,4);      clear FF_crop;
        
        [JPEG_normd_input] = concatenate_data(JPEG_normd,1); clear JPEG_normd;
        [JP2K_normd_input] = concatenate_data(JP2K_normd,1); clear JP2K_normd;
        [WN_normd_input] = concatenate_data(WN_normd,1);     clear WN_normd;
        [BLUR_normd_input] = concatenate_data(BLUR_normd,1); clear BLUR_normd;
        [FF_normd_input] = concatenate_data(FF_normd,1);     clear FF_normd;
        
        [JPEG_dmos_corr] = concatenate_data(JPEG_dmos,1);    clear JPEG_dmos;
        [JP2K_dmos_corr] = concatenate_data(JP2K_dmos,1);    clear JP2K_dmos;
        [WN_dmos_corr] = concatenate_data(WN_dmos,1);        clear WN_dmos;
        [BLUR_dmos_corr] = concatenate_data(BLUR_dmos,1);    clear BLUR_dmos;
        [FF_dmos_corr] = concatenate_data(FF_dmos,1);        clear FF_dmos;
 
        % This is just to fix the indexes for ablation and study tests

        [trainIdxs_JPEG, testIdxs_JPEG,  trainJPEG_IN,  trainJPEG_OUT,  testJPEG_IN] = Divide_random_kfold(JPEG_crop_input, JPEG_normd_input,k); %clear JPEG_crop_input JPEG_normd_input;
        [trainIdxs_JP2K, testIdxs_JP2K,  trainJP2K_IN,  trainJP2K_OUT,  testJP2K_IN] = Divide_random_kfold(JP2K_crop_input, JP2K_normd_input,k); %clear JP2K_crop_input JP2K_normd_input;
        [trainIdxs_WN,   testIdxs_WN,    trainWN_IN,    trainWN_OUT,    testWN_IN]   = Divide_random_kfold(WN_crop_input,   WN_normd_input,k);   %clear WN_crop_input WN_normd_input;
        [trainIdxs_BLUR, testIdxs_BLUR,  trainBLUR_IN,  trainBLUR_OUT,  testBLUR_IN] = Divide_random_kfold(BLUR_crop_input, BLUR_normd_input,k); %clear BLUR_crop_input BLUR_normd_input;
        [trainIdxs_FF,   testIdxs_FF,    trainFF_IN,    trainFF_OUT,    testFF_IN]   = Divide_random_kfold(FF_crop_input,   FF_normd_input,k);   %clear FF_crop_input FF_normd_input;
        
        for i=1:k
            
            trainMatrix_IN{i}  = cat(4, trainJPEG_IN{i},  trainJP2K_IN{i},  trainWN_IN{i},  trainBLUR_IN{i},  trainFF_IN{i}); %Inout and output for the network 80%
            trainMatrix_OUT{i} = cat(1, trainJPEG_OUT{i}, trainJP2K_OUT{i}, trainWN_OUT{i}, trainBLUR_OUT{i}, trainFF_OUT{i});
            
            %testMatrix_IN{i}  = cat(4, testJPEG_IN{i},  testJP2K_IN{i}, testWN_IN{i}, testBLUR_IN{i}, testFF_IN{i});  %Input for Testing 20%
            %testMatrix_OUT{i} = cat(1, testJPEG_OUT{i}, testJP2K_OUT{i}, testWN_OUT{i}, testBLUR_OUT{i}, testFF_OUT{i});
            
            %testIdxs{i} = cat(1, testIdxs_WN{i}, testIdxs_JP2K{i}, testIdxs_JPEG{i}, testIdxs_BLUR{i}, testIdxs_FF{i});
            %Subj_S{i} = [Dmos_new(testIdxs{i})];
        end
        
        %% Initialization
        N_F = 128; % Number of Features to be extracted from each patch
        Epoch = 50; %50
        M_B = 64; %60
        
        
        %layersTransfer = net_all.Layers(1:end);
        Score_dmos = []; DATA_C = []; DATA_F = []; OB = []; SB = [];
        TEST_WN_all = []; TEST_FF_all = []; TEST_BLUR_all = []; TEST_JPEG_all = []; TEST_JP2k_all = [];
        DATA_WN = []; DATA_JPEG  = [];    DATA_JP2k = [];  DATA_BLUR  = [];  DATA_FF = [];
        %All_NET= []; TEST_NET = []; DATA = [];
        net = []; net_help = vgg16;
              layers = [
                imageInputLayer([32 32 3],"Name","imageinput")
                net_help.Layers(2:end-9)
           
                
                fullyConnectedLayer(N_F,"Name","fc_1")
                reluLayer("Name","relu4")
                
                fullyConnectedLayer(10,"Name","fc_2")
                %reluLayer("Name","relu5")
                
                fullyConnectedLayer(1,"Name","fc_3")
                
                regressionLayer("Name","regressionoutput")];
            
            %         'Plots','training-progress'
            %         'Shuffle','every-epoch'
            
            options = trainingOptions('sgdm',...
                'LearnRateSchedule', 'piecewise',...
                'LearnRateDropFactor', 0.9,...
                'LearnRateDropPeriod', 15,...
                'MiniBatchSize',M_B,...
                'MaxEpochs',Epoch, ...
                'Shuffle','every-epoch',...
                'InitialLearnRate',1e-2);
        %% Train the network for k times (Train on 80% and test on the rest 20%
        % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
        for i = 1:k
         
            net = trainNetwork(trainMatrix_IN{i},trainMatrix_OUT{i},layers,options)
            %net = trainNetwork(augimds,layers,options)
            %net.Layers
            %Models{index,m} = net; % save the model
            
            TEST_WN{i}   = predict(net,testWN_IN{i});
            TEST_JPEG{i} = predict(net,testJPEG_IN{i}); %compute and stock results
            TEST_JP2k{i} = predict(net,testJP2K_IN{i});
            TEST_BLUR{i} = predict(net,testBLUR_IN{i});
            TEST_FF{i}   = predict(net,testFF_IN{i});
            
            TEST_WN_all   = [TEST_WN_all;   TEST_WN{i}];
            TEST_JP2k_all = [TEST_JP2k_all; TEST_JP2k{i}];
            TEST_JPEG_all = [TEST_JPEG_all; TEST_JPEG{i}];
            TEST_BLUR_all = [TEST_BLUR_all; TEST_BLUR{i}];
            TEST_FF_all   = [TEST_FF_all;   TEST_FF{i}];
            % Score_dmos = [Score_dmos; Subj_S{i}]; % Stock test dmos for re-ordering later.
            % DATA = [DATA; testIdxs{i}]; % Stock the test indexes for re-ordering later.
            DATA_WN   = [DATA_WN; testIdxs_WN{i}];
            DATA_JPEG = [DATA_JPEG; testIdxs_JPEG{i}];
            DATA_JP2k = [DATA_JP2k; testIdxs_JP2K{i}];
            DATA_BLUR = [DATA_BLUR; testIdxs_BLUR{i}];
            DATA_FF   = [DATA_FF; testIdxs_FF{i}];
            m = m + 1;
        end
        
        %% Re-order the Dmos and do the Average of prediction
        
        
        
        %
        for j=1:length(TEST_WN_all)
            
            DATA_WN_C(DATA_WN(j)) = [TEST_WN_all(j)]; %re-order the dmos score based on indexes from 1:end
            % Score_dmos_F(DATA(j)) = [Score_dmos(j)];
        end
        
        for j=1:length(TEST_JP2k_all)
            
            DATA_JP2k_C(DATA_JP2k(j)) = [TEST_JP2k_all(j)];
            % Score_dmos_F(DATA(j)) = [Score_dmos(j)];
        end
        
        for j=1:length(TEST_JPEG_all)
            
            DATA_JPEG_C(DATA_JPEG(j)) = [TEST_JPEG_all(j)];
            % Score_dmos_F(DATA(j)) = [Score_dmos(j)];
        end
        
        for j=1:length(TEST_BLUR_all)
            
            DATA_BLUR_C(DATA_BLUR(j)) = [TEST_BLUR_all(j)];
            % Score_dmos_F(DATA(j)) = [Score_dmos(j)];
        end
        
        for j=1:length(TEST_FF_all)
            
            DATA_FF_C(DATA_FF(j)) = [TEST_FF_all(j)];
            % Score_dmos_F(DATA(j)) = [Score_dmos(j)];
        end
        
        
        DATA_C = [DATA_WN_C'; DATA_JP2k_C'; DATA_JPEG_C'; DATA_BLUR_C'; DATA_FF_C'];
        Score_dmos_F = [WN_dmos_corr; JP2K_dmos_corr; JPEG_dmos_corr; BLUR_dmos_corr; FF_dmos_corr];
        P = 0;
        for j=1:365        % compute the mean score from patches score
            
            %M = size(Stock_img_new{j},4);
            M = Patches_nbr(j);
            DATA_F(j) = mean(DATA_C(P+1:P+M));    % The mean score of each stereo image
            dmos_F(j) = mean(Score_dmos_F(P+1:P+M)); % Just to be sure that computational is correct- this should match the Dmos if the database.
            P = P + M;
        end
        
        FINAL_score = zeros(1,365);
        FINAL_dmos = zeros(1,365);
        
        FINAL_score = Combine_distortions_L1(DATA_F);
        FINAL_dmos  = Combine_distortions_L1(dmos_F); % Just to make sure that everything is OK This should match the Dmos of DB
        %% Evaluate the Net_Model
        
        
        disp('Results of Deep-Saliency LIVE I>>>');
        SB = dmos;               % Subjective Score
        OB = FINAL_score';        % Objective Score
        %figure,
        [Srocc,Krooc, cc,rmse] = logistic_distortions_cc_L1(double(OB),double(SB))
        
        %% Stock the results for mutliple iterations
        Score_Stock{R} = FINAL_score';
        Final_R(R,:) = [Srocc,Krooc, cc,rmse];
        Average_R = mean(Final_R);
        disp(Average_R);
        Saliency_vs_Score{index}= {Final_R;Average_R;Sal_Value;size(DATA_C);FINAL_score'};
        Performance(index,1:4) = Average_R(:);
        X(index) = [Sal_Value];
        Y(index) = [Average_R(1)];
        
        %save(['Idxs_L1_', num2str(R),'-',num2str(Sal_Value),'.mat'],'trainIdxs_JPEG', 'testIdxs_JPEG', 'trainIdxs_JP2K', 'testIdxs_JP2K', 'trainIdxs_WN', 'testIdxs_WN', 'trainIdxs_BLUR', 'testIdxs_BLUR', 'trainIdxs_FF', 'testIdxs_FF');
        clearvars -except m k Performance Sal_Value Stock_normdL1 Stock_Dmos_new Stock_img Final_R Percentage index FINAL_score Average_R Saliency_vs_Score Models dmos norm_dL1 Cyclopean_L1 Saliency_3d_L1 X Y
    end
    
    index = index + 1;
    % Here you can save the indexes, we therefore will be using the same
    % random split train/test images for each run. It needed for ablation
    % study.
    %save(['trainIdxs_L2_', num2str(R),'-',num2str(Sal_Value),'.mat'],'trainIdxs_JPEG', 'testIdxs_JPEG', 'trainIdxs_JP2K', 'testIdxs_JP2K', 'trainIdxs_WN', 'testIdxs_WN', 'trainIdxs_BLUR', 'testIdxs_BLUR', 'trainIdxs_FF', 'testIdxs_FF');
    
end
plot(X,Y,'k--*','LineWidth',2), grid on
xlabel('Saliency Values'),
ylabel('SROCC LIVE I'),
hold on
str = num2str(Performance);
text(X,Y,str,'Color','r','FontSize',8);
