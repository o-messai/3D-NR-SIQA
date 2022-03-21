function [ disp_comp_L ] = mj_GenMergeView( imL,imR,Dmap_L )

%   Generate merged view from disparity map

     S_L=imL; % greate buffer for systhesized view
      S_L(:)=0; % reset
      M_L=S_L;
      sz=size(imL);
%      [Fx,Fy] = gradient(double(imL));
%      gdL= sqrt(Fx.^2+Fy.^2);
%      
%      [Fx,Fy] = gradient(double(imR));
%      gdR= sqrt(Fx.^2+Fy.^2);
%     sm=5^-30;
%     winsize=64;
%     win=fspecial('gaussian',winsize,winsize/4);%    % win=ones(winsize,winsize)./double(winsize*winsize);    
%     gdL=gdL+1;
%     gdR=gdR+1;
% 
%      gdL   = filter2(win, gdL, 'same');
%      gdR   = filter2(win, gdR, 'same');   
%      gdS = gdL+gdR+sm;
%      gdL = gdL./gdS;
%      gdR = gdR./gdS;
     for x=1:sz(1)
         for y=1:sz(2)
             idxNew = y- Dmap_L(x,y);
             idxNew = max(1,idxNew);
             idxNew = min(sz(2)-1,idxNew);
             S_L(x,y)= (idxNew- floor(idxNew))*imR(x,floor(idxNew)+1)+(floor(idxNew)+1-idxNew)*imR(x,floor(idxNew)); % interpolated between pixels
                          
            M_L(x,y)= (imL(x,y)+S_L(x,y))*0.5; %% the key is how to merge two view, for now just average (it is certainly wrong) 0601/2011 Ming
        %     M_L(x,y) = (imL(x,y)*gdL(x,y)+S_L(x,y)*gdR(x,y));
             % M_L(x,y)= (imL(x,y)+imR(x,y))*0.5; %% the key is how to merge two view, for now just average (it is certainly wrong) 0601/2011 Ming             
            % M_L(x,y)= max(imL(x,y),S_L(x,y));          
         end
     end
     
end

