
function [M_W_D,dmap_test, GBD_L, GBD_R] = MJ3DQA(imDL,imDR,max_disp)




% Input
% imRL - ref_left view
% imRR - ref_right view
% imDL - test_left view
% imDR - test_right view
% max_disp  - max disparity value . This value may be tuned for different dataset.  

% Output
% score- Predict QA score
% dmap_ref - estimated disparity from reference pair
% dmap_test - estimated disparity from test pair

if (nargin < 2)
    score = -Inf;
    disparity_map = 0;
    return;
end
if (nargin ==2 )
    max_disp = 25;  % the dault value is set based on experiments on LIVE 3D IQA database
end


% if(size(imDL,3)==3)
%    
%     imDL = rgb2gray(imDL);
%     imDR = rgb2gray(imDR);
%     
% end


imsz = size(imDL);


[fdsp, dmap_test, confidence, diff] = mj_stereo_SSIM(imDL,imDR, max_disp);
[ disp_comp_dl] = mj_computeDispCompIm( imDL,imDR,dmap_test );


[D_L_Gabor_RS,D_L_Gabor_Bound]=ExtractGaborResponse(imDL);
[Syn_D_Gabor_RS,Syn_D_Gabor_Bound]=ExtractGaborResponse(imDR);


SL_en=zeros(imsz(1),imsz(2),2); %4-scales
SS_en=zeros(imsz(1),imsz(2),2); %4-scales
for mm=1:4
    SL_en(:,:,mm) = D_L_Gabor_RS{2+mm,1}+D_L_Gabor_RS{2+mm,2}+D_L_Gabor_RS{2+mm,3}+D_L_Gabor_RS{2+mm,4}+D_L_Gabor_RS{2+mm,5}+D_L_Gabor_RS{2+mm,6}+D_L_Gabor_RS{2+mm,7}+D_L_Gabor_RS{2+mm,8};
    SS_en(:,:,mm) = Syn_D_Gabor_RS{2+mm,1}+Syn_D_Gabor_RS{2+mm,2}+Syn_D_Gabor_RS{2+mm,3}+Syn_D_Gabor_RS{2+mm,4}+Syn_D_Gabor_RS{2+mm,5}+Syn_D_Gabor_RS{2+mm,6}+Syn_D_Gabor_RS{2+mm,7}+Syn_D_Gabor_RS{2+mm,8};;
end
GBD_L = SL_en(:,:,1) ; %3-4-5-6
GBD_R = SS_en(:,:,1) ;


[ disp_comp_GBdl ] = mj_computeDispCompIm( GBD_L,GBD_R,dmap_test );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[ M_W_D ] = mj_GenMergeWEntropy( imDL,imDR,GBD_L,GBD_R );
%[ M_W_D ] = mj_GenMergeWEntropy( imDL,imDR,GBD_L,GBD_R );
[ M_W_D ] = mj_GenMergeWEntropy( imDL,disp_comp_dl,GBD_L,disp_comp_GBdl );
 %[ M_W_D ] = mj_GenMergeWEntropy( imDL,imDR,GBD_L,disp_comp_GBdl );


% [ GB_W_D, syn_GBd ] = mj_GenMergeView( GBD_L,GBD_R,Dmap_L );
 %[ M_W_D, syn_d ] = mj_GenMergeView( imDL,imDR,Dmap_L );
end




