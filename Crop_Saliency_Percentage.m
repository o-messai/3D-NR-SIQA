function [img_cropped, S] = Crop_Saliency_Percentage(Cyclopean, Saliency, C_x, C_y, Thershold, Percentage)
% C_x, C_y select the pixels patch
%img_cropped = gpuArray([]);
Marked_pixel = gpuArray([]);  % Run on GPU

[X, Y] = find(Saliency>=Thershold); %find the pixels x, y values with respect to saliency.
Add_x = 0; Add_y = 0; m =2;

[W] = size(Cyclopean,1); [L] = size(Cyclopean,2);
W=W/2; 
L=L/2;
p=floor(C_x/2);
k=floor(C_y/2);

img_cropped(:,:,:,1) = Cyclopean(W-p:W+p+1,L-k:L+k+1,:); % crop the centre as the first saliency patch
Marked_pixel = [W-p:W+p+1; L-k:L+k+1]';
Cyclopean_temp = Cyclopean;
for i=1:length(X)-1
    
    Differenc_x = X(i+1)-X(i);
    if Differenc_x==1
        Add_x = Add_x + 1;
    else
        Add_x = 0;
    end
    
    if Add_x == C_x
        Add_x = 0;
        
        Point_x = i-C_x;
        
        if Point_x<=0   % indexe limits for Y(j)
            Point_x=1;
        end
        
        for j=Point_x:i
            
            Differenc_y = Y(j+1)-Y(j);
            
            if Differenc_y==0
                Add_y = Add_y + 1;
            else
                Add_y = 0;
            end
            
            if Add_y == C_y
                Add_y = 0;
                
                Start_x = X(i)-C_x; Stop_x = X(i);
                Start_y = Y(j)-C_y; Stop_y = Y(j);
                
                if Start_x <= 0
                    Start_x = 1;
                    Stop_x = C_x+1; % Image limits in the x direction
                end
                if Start_y <= 0
                    Start_y = 1;
                    Stop_y = C_y+1; % Image limits in the y direction
                end
                
                
                %row_zeros_p = find(all(Marked_pixel == [Start_x Start_y],2), 1);
                %row_zeros_x = find(all(Marked_pixel == [Stop_x Start_y],2), 1);
                
                [WW , ~] = find(Cyclopean_temp(Start_x:Stop_x,Start_y:Stop_y,1) == 255);
                
                N_found = length(WW);
                N_Per = (Percentage*C_x*C_y)/100;
                
                %if isempty(row_zeros_p) && N_found<=N_Per  %&& isempty(row_zeros_x)
                if N_found<=N_Per    
                    img_cropped(:,:,:,m) = Cyclopean(Start_x:Stop_x,Start_y:Stop_y,:);
                    Cyclopean_temp(Start_x:Stop_x,Start_y:Stop_y,:) = 255;
                    
                    %subplot(2,2,1), imshow(uint8(Cyclopean_temp));
                    %subplot(2,2,2), imagesc(Saliency);
                    %subplot(2,2,3), imshow(uint8(img_cropped(:,:,:,m)));
                    %subplot(2,2,4), imshow(img_cropped(:,:,:,m));
                   
                    m = m + 1;
                    %pause(1);
                    ind = 0;
                for w = 1:C_x
                    Cropped_pixel(:,:) = [repmat(Start_x+ind,C_x+1,1)'; Start_y:Stop_y];
                    %Marked_pixel = [Marked_pixel; Cropped_pixel'];
                    %disp(size(Marked_pixel));
                    ind = ind +1;
                end
                end
          
                
            end
            
        end
        %i = i + C_x;
    end
    %disp(size(img_cropped));
    S = size(img_cropped,4);
end



