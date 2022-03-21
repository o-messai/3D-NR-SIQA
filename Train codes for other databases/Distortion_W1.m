% =================================================================
%  University of Constantine-1
%  Automatic and Robotic Laboratory
%  Copyright(c) 2017  MESSAI Oussama
%  e-mail: mr.oussama.messai@gmail.com 
%  All Rights Reserved.

% -----------------------------------------------------------------
clc; %clear  %clean screen
close all;
addpath( genpath('Files mat') )
load('IVCmosPh1.mat'); 
load('MOS_W1.mat'); % load database information and DMOS data

SBwn = [], OBwn = [], SBgb = [], OBgb = [], SBjpeg = [], OBjpeg = [];
OB = double(Predicted);
for iPoint = 1:330
    
%White noise    

F = strfind(MOS_W1{iPoint,1},'W');
if ~isempty(F)
SBwn = [SBwn IVCmosPh1(iPoint)];       % Subjective Score
OBwn = [OBwn OB(iPoint)];             % Objective Score
end

%Guassian noise

F = strfind(MOS_W1{iPoint,1},'G');
if ~isempty(F)
SBgb = [SBgb IVCmosPh1(iPoint)];       % Subjective Score
OBgb = [OBgb OB(iPoint)];             % Objective Score
end
%JPEG

F = strfind(MOS_W1{iPoint,1},'J');
if ~isempty(F)
SBjpeg = [SBjpeg IVCmosPh1(iPoint)];       % Subjective Score
OBjpeg = [OBjpeg OB(iPoint)];             % Objective Score
end
end

metric_1 = corr(SBwn, OBwn, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SBwn, OBwn, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SBwn, OBwn, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics_wn = [metric_1;metric_2;metric_3]

figure,
[rocc,Krooc, cc,rmse] = logistic_cc(OBwn,SBwn)
legend(' White noise','Curve fitted with logistic function');

%%

metric_1 = corr(SBgb, OBgb, 'type', 'pearson');     
metric_2 = corr(SBgb, OBgb, 'type', 'spearman');    
metrics_gb = [metric_1;metric_2;metric_3]

figure,
[rocc,Krooc, cc,rmse] = logistic_cc(OBgb,SBgb)
legend(' Guassian noise','Curve fitted with logistic function');
%%

metric_1 = corr(SBjpeg, OBjpeg, 'type', 'pearson');     
metric_2 = corr(SBjpeg, OBjpeg, 'type', 'spearman');    
metric_3 = corr(SBjpeg, OBjpeg, 'type', 'kendall');    
metrics_jpeg = [metric_1;metric_2;metric_3]

figure,
[rocc,Krooc, cc,rmse] = logistic_cc(OBjpeg,SBjpeg)
legend(' JPEG','Curve fitted with logistic function');

