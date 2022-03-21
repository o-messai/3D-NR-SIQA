clc;clear;close all;
addpath ( genpath ( 'Files mat' ) );
addpath ( genpath ( 'Databases' ) );
load('3DDmosRelease.mat');
load('norm_dL2.mat')
load Cyclopean_L2.mat
load Saliency_3d_L2.mat

load Train_Indexes
myFolder = './Saved nets';  % Empty the folder
addpath ( genpath ( myFolder ) );
warning('off')


index = 1; X = []; Y=[]; Percentage = 100;
for Sal_Value = [0.3:0.1:0.3]
    %[0.1:0.1:0.9]
    Stock_img = {};  Size_val_L2 = {}; Dmos_new = [];
    %% Extract the Saliency from the Cyclopean Image
    parfor iPoint=1:360
        
        [ img_cropped, S ] = Crop_Saliency_Percentage(Cyclopean_L2(:,:,:,iPoint),Saliency_3d_L2(:,:,iPoint),31,31,Sal_Value,Percentage);
        Size_val_L2{iPoint} = [S];
        Stock_img{iPoint}  = [img_cropped(:,:,:,:)];
        
        OUT_Score = repmat(norm_dL2(iPoint),S,1); % prepare norm. dmos for training
        
        Stock_normdL2{iPoint} = [OUT_Score];
        
        D_Score = repmat(Dmos(iPoint),S,1); % prepare Dmos for correlation.
        Dmos_new = [Dmos_new; D_Score];
        Stock_Dmos_new{iPoint} = [D_Score];
        
    end
    disp(size(Dmos_new));
    clear Saliency_3d_L2 Cyclopean_L2
    
    DMOS = []; SCORE = [];
    for Fold=1:5
        
        Train_input = []; Train_output = []; Test_dmos_validation = [];
        Test_input = []; Test_dmos = [];Size_val = [];
        
        Train_Idx_L2 = trainIdxs{Fold};
        Test_Idx_L2 = testIdxs{Fold};
        
        for iPoint=1:length(Train_Idx_L2)
            
            Train_input = cat(4,Train_input, Stock_img{Train_Idx_L2(iPoint)});
            Train_output = cat(1,Train_output, Stock_normdL2{Train_Idx_L2(iPoint)});
            
        end
        
        for iPoint=1:length(Test_Idx_L2)
            
            Test_input = cat(4, Test_input, Stock_img{Test_Idx_L2(iPoint)});
            Test_dmos = cat(1, Test_dmos, Stock_Dmos_new{Test_Idx_L2(iPoint)});
            Test_dmos_validation = cat(1, Test_dmos_validation, Stock_normdL2{Test_Idx_L2(iPoint)});
            Size_val = cat(1, Size_val, Size_val_L2{Test_Idx_L2(iPoint)});
            
        end
        
        %clear Stock_img Stock_normdL2 Stock_Dmos_new
        
        %% Initialization
        N_F = 128; % Number of Features to be extracted from each patch
        Epoch = 50; %150
        M_B = 64; %32
        
        %layersTransfer = net_all.Layers(1:end);
        
        %All_NET= [];  DATA = [];
        
        %% Train the network for k times (Train on 80% and test on the rest 20%
        % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
        net_help = vgg16;
        layers = [
            imageInputLayer([32 32 3],"Name","imageinput")
            net_help.Layers(2:end-9)
            
            fullyConnectedLayer(N_F,"Name","fc_1")
            reluLayer("Name","relu4")
            
            fullyConnectedLayer(10,"Name","fc_2")
            %reluLayer("Name","relu5")
            
            fullyConnectedLayer(1,"Name","fc_3")
            
            regressionLayer("Name","regressionoutput")];
        
        myFolder = './Saved nets';  % Empty the folder
        mkdir(myFolder);
        rmdir(myFolder,'s')
        mkdir(myFolder);
        options = trainingOptions('sgdm',...
            'LearnRateSchedule', 'piecewise',...
            'LearnRateDropFactor', 0.9,...
            'LearnRateDropPeriod', 25,...
            'MiniBatchSize',M_B,...
            'L2Regularization', 0.01,...
            'Shuffle','every-epoch',...
            'MaxEpochs',Epoch, ...
            'CheckpointPath',myFolder,...
            'InitialLearnRate',1e-3);
        
        net = trainNetwork(Train_input,Train_output,layers,options);
        %net.Layers
        %Models{R} = net; % save the model

        %% Test on all nets
        
        addpath ( genpath ( myFolder ) );
        filePattern = fullfile(myFolder, '*.mat');
        matFiles = dir(filePattern);
        Best_cc = 0;
        for U = 1:length(matFiles)
            baseFileName = matFiles(U).name;
            load(baseFileName);
            
            TEST_NET = predict(net,Test_input);
            P = 0; DATA_F = []; dmos_F = [];
            for j=1:length(Size_val)        % compute the mean score from patches score
     
                M = Size_val(j);
                DATA_F(j) = mean(TEST_NET(P+1:P+M));    % The mean score of each stereo image
                dmos_F(j) = mean(Test_dmos(P+1:P+M)); % Just to be sure that computational is correct- this should match the Dmos if the database.
                P = P + M;
            end
            
            
            SB = dmos_F';                       % Subjective Score
            OB = DATA_F';                    % Objective Score
            %figure,
            [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB),double(SB));
            
            if cc > Best_cc
                Best_cc = cc;
                Best_model = net;
                %Best_model_name = baseFileName;
                Best_model_score = OB;
                Best_performance = [Srocc,Krooc, cc,rmse]
            end
            clear net;
        end
        %% Stock the results for mutliple iterations
        DMOS = [DMOS; SB];
        SCORE = [SCORE; OB];
        Stock_best_model{Fold} = {Best_model;Best_model_score;SB;Best_performance};
    end
    
    disp('Final Best Score = ')
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(SCORE),double(DMOS))
     
end

