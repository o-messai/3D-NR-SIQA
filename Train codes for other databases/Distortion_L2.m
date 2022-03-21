% =================================================================
%  University of Constantine-1
%  Automatic and Robotic Laboratory
%  Copyright(c) 2017  MESSAI Oussama
%  e-mail: mr.oussama.messai@gmail.com 
%  All Rights Reserved.

% -----------------------------------------------------------------
clc;  %clean screen
close all;
addpath ( 'Files mat' )
load('3DDmosRelease.mat');   % load database information and DMOS data
%load('OBd.mat'); %load Objective Score
Score = double(Predicted);
%White noise

SBwn = Dmos([1:9 46:54 91:99 136:144 181:189 226:234 271:279 316:324]);       % Subjective Score
OBwn = Score([1:9 46:54 91:99 136:144 181:189 226:234 271:279 316:324]) ;      % Objective Score

metric_1 = corr(SBwn, OBwn, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SBwn, OBwn, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SBwn, OBwn, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics_wn = [metric_1;metric_2;metric_3]

figure,
[Srocc,Krooc, Lcc,rmse] = logistic_cc(OBwn,SBwn)
legend(' White noise','Curve fitted with logistic function');


%JP2K

SBjp2 = Dmos([10:18 55:63 100:108 145:153 190:198 235:243 280:288 325:333]);       
OBjp2 = Score([10:18 55:63 100:108 145:153 190:198 235:243 280:288 325:333]) ;     

metric_1 = corr(SBjp2, OBjp2, 'type', 'pearson');    
metric_2 = corr(SBjp2, OBjp2, 'type', 'spearman');    
metric_3 = corr(SBjp2, OBjp2, 'type', 'kendall');     
metrics_jp2 = [metric_1;metric_2;metric_3]

figure,
[Srocc,Krooc, Lcc,rmse] = logistic_cc(OBjp2,SBjp2)
legend(' JP2K','Curve fitted with logistic function');



%JPEG

SBjpeg = Dmos([19:27 64:72 109:117 154:162 199:207 244:252 289:297 334:342]);      
OBjpeg = Score([19:27 64:72 109:117 154:162 199:207 244:252 289:297 334:342]) ;      

metric_1 = corr(SBjpeg, OBjpeg, 'type', 'pearson');     
metric_2 = corr(SBjpeg, OBjpeg, 'type', 'spearman');    
metric_3 = corr(SBjpeg, OBjpeg, 'type', 'kendall');    
metrics_jpeg = [metric_1;metric_2;metric_3]

figure,
[Srocc,Krooc, Lcc,rmse] = logistic_cc(OBjpeg,SBjpeg)
legend(' JPEG','Curve fitted with logistic function');



%Guassian noise

SBgb = Dmos([28:36 73:81 118:126 163:171 208:216 253:261 298:306 343:351]);       
OBgb = Score([28:36 73:81 118:126 163:171 208:216 253:261 298:306 343:351]) ;      

metric_1 = corr(SBgb, OBgb, 'type', 'pearson');     
metric_2 = corr(SBgb, OBgb, 'type', 'spearman');    
metrics_gb = [metric_1;metric_2;metric_3]

figure,
[Srocc,Krooc, Lcc,rmse] = logistic_cc(OBgb,SBgb)
legend(' Guassian noise','Curve fitted with logistic function');


%Fast fading

SBff = Dmos([37:45 82:90 127:135 172:180 217:225 262:270 307:315 352:360]);       
OBff = Score([37:45 82:90 127:135 172:180 217:225 262:270 307:315 352:360]) ;      

metric_1 = corr(SBff, OBff, 'type', 'pearson');     
metric_2 = corr(SBff, OBff, 'type', 'spearman');    
metric_3 = corr(SBff, OBff, 'type', 'kendall');     
metrics_ff = [metric_1;metric_2;metric_3]

figure,
[Srocc,Krooc, Lcc,rmse] = logistic_cc(OBff,SBff)
legend(' Fast fading','Curve fitted with logistic function');