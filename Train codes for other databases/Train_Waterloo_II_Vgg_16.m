clear;clc;close all;
addpath ( genpath ( 'Files mat' ) );
load('IVCmosPh2.mat');
load('norm_dW2.mat')
%load Cyclopean_W2.mat
load Saliency_3d_W2.mat
warning('off')

matObj = matfile('Cyclopean_W2.mat');
index = 1; X = []; Y=[]; Size_val = [];
for Sal_Value = [0.3:0.1:0.3]
T_input = []; T_output = []; Dmos_new = []; Stock_img = {};

%% Extract the Saliency from the Cyclopean Image
for iPoint=1:460
    
    [ img_cropped, S ] = Crop_Saliency_Percentage(matObj.Cyclopean_W2(:,:,:,iPoint),Saliency_3d_W2(:,:,iPoint),63,63,Sal_Value,85);        
    %Stock_img{iPoint} = [img_cropped(:,:,:,:)];
    
    Size_val = [Size_val S];
    OUT_Score = repmat(norm_dW2(iPoint),S,1); % prepare norm. dmos for training
    T_output = [T_output;OUT_Score];
    Stock_normdW2{iPoint} = [OUT_Score];
    
    D_Score = repmat(IVCmosPh2(iPoint),S,1); % prepare Dmos for correlation.
    Dmos_new = [Dmos_new; D_Score];
    Stock_Dmos_new{iPoint} = [D_Score];   
    T_input = cat(4,T_input, img_cropped); % For training input
    disp(iPoint);
    
    
end
disp(size(Dmos_new)); disp(Sal_Value);

clear Cyclopean_W2 Saliency_3d_W2


repeat = 1; 
for R=1:repeat
    
    %% Initialization
    N_F = 128; % Number of Features to be extracted from each patch
    Epoch = 50; %150
    M_B = 64; %60

    
    %layersTransfer = net_all.Layers(1:end);
    Score_dmos = [];
    All_NET= []; TEST_NET = []; DATA = [];
    k = 5;
    %% Divide dataset to 80%-20% Randomly
     net = []; net_help = vgg16;
              layers = [
                imageInputLayer([64 64 3],"Name","imageinput")
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
      
        
    
    cv = cvpartition(length(T_output), 'kfold',k);
    
    for i=1:k
        
        trainIdxs{i} = find(training(cv,i));  %trainIdxs{i} = find(ind_train(:,i)==1);
        testIdxs{i}  = find(test(cv,i));      %testIdxs{i}  = find(ind_test(:,i)==1);
        
        trainMatrix_IN  = [T_input(:,:,:,trainIdxs{i})]; %Inout and output for the network 80%
        trainMatrix_OUT = [T_output(trainIdxs{i})];
        
        testMatrix_IN  = [T_input(:,:,:,testIdxs{i})];  %Input for Testing 20%
        %testMatrix_OUT = [T_output(testIdxs{i})];
        
        %Subjectiv_S{i} = [Dmos(testIdxs{i})];
        Subjectiv_S = [Dmos_new(testIdxs{i})];
        
         net = trainNetwork(trainMatrix_IN,trainMatrix_OUT,layers,options)
        %net.Layers
       % Models{1,i} = net; % save the model
   
        TEST_NET = predict(net,testMatrix_IN); %compute and stock results
        All_NET = [All_NET; TEST_NET];
        
        Score_dmos = [Score_dmos; Subjectiv_S]; % Stock test dmos for re-ordering later.
        DATA = [DATA; testIdxs{i}]; % Stock the test indexes for re-ordering later.
    
        
    end
    
    %% Train the network for k times (Train on 80% and test on the rest 20%
    % Transfer Learning use  : layersTransfer = alexnet.Layers(1:end-3);
   
        
       
    
    %% Re-order the Dmos and do the Average of prediction
    
              
    for j=1:length(All_NET)
        
       DATA_C(DATA(j)) = [All_NET(j)]; %re-order the dmos score based on indexes from 1:end  
       Score_dmos_F(DATA(j)) = [Score_dmos(j)];
       
       
    end
    P = 0;
    for j=1:460        % compute the mean score from patches score
        
        M = Size_val(j);      
        DATA_F(j) = mean(DATA_C(P+1:P+M));    % The mean score of each stereo image
        dmos_F(j) = mean(Score_dmos_F(P+1:P+M)); % Just to be sure that computational is correct- this should match the Dmos if the database.       
        P = P + M;
    end
    
    %% Evaluate the Net_Model
    
    
    disp('Results of Deep-Saliency Waterloo II>>>');
    SB = IVCmosPh2;               % Subjective Score 
    OB = DATA_F';                    % Objective Score
    %figure,
    [Srocc,Krooc, cc,rmse] = logistic_cc(double(OB),double(SB))
    
   %% Stock the results for mutliple iterations
     
        FINAL_score = DATA_F;
        Score_Stock{R} = FINAL_score';
        Final_R(R,:) = [Srocc,Krooc, cc,rmse];
        Average_R = mean(Final_R);
        disp(Average_R);
        Saliency_vs_Score{index}= {Final_R;Average_R;Sal_Value;size(DATA_C);FINAL_score'};
        Performance(index,1:4) = Average_R(:);
        X(index) = [Sal_Value];
        Y(index) = [Average_R(1)];
    
end
        
        index = index + 1;

end

plot(X,Y,'k--*','LineWidth',2), grid on
xlabel('Saliency Values'),
ylabel('SROCC Waterloo II'),
hold on
str = num2str(Performance);
text(X,Y,str,'Color','r','FontSize',8);
