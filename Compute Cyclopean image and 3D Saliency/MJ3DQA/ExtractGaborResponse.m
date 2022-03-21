function [ image_filtered_ori, image_boundary_ori ] = ExtractGaborResponse( IM )
%UNTITLED Summary of this function goes here
% Detailed explanation goes here
%Author Che-Chun Su
load Gabor_no_DC_unit_energy_07_octave.mat;

number_fr = size ( gabor_fr(:) , 1 );
number_or = size ( gabor_or(:) , 1 );
% parameters for divisive normalization ------
sigma = 0.01; % semi-saturation constant? or just to avoid zero denominator? or 0.01 (1% noise)?
if_fr_parent = 12;
if_or_neighbor = 1;
spatial_size_x = 3;
spatial_size_y = 3;
spatial_x = (spatial_size_x-1)/2;
spatial_y = (spatial_size_y-1)/2;
spatial_size = spatial_size_x*spatial_size_y;
% ------


image_filtered_ori = cell ( number_fr , number_or );
image_boundary_ori = zeros ( number_fr , number_or );

  for fr =1:number_fr
        for or =1:1:number_or
            
            % ------
            %disp ( sprintf ( 'Filtering image... (  frequency: %d , orientation: %d )'  , fr , or ) );
            % ------
            
            gabor_complex = gabor_cosine{fr,or} + i*gabor_sine{fr,or};
            image_filtered(:,:) = filter2 ( gabor_complex , double(IM(:,:)) );

            image_boundary = (size ( gabor_complex , 1 ) - 1)/2;
            
       
            image_filtered_ori{fr,or} = abs(image_filtered);
            image_boundary_ori(fr,or) = image_boundary;
            
%             filename = sprintf ( output_filename_1 , s , fr , or );
%             save ( filename , 'image_filtered' , 'image_boundary' );
           %  figure,
			% imshow(image_filtered);
            clear image_filtered;
            
        end        
    end
end

