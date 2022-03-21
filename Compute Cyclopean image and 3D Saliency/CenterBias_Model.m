
%% Generate the saliency maps with only a gaussian kernel for the center bias factor

function cbm = CenterBias_Model(dep_map_in, deg)

% deg = 2.5;
%% the following parameters are set by display sizes and viewing distance
    sigma_x = tan(deg/180*pi) * 970 / (310/1080) ;

    r = size(dep_map_in,1);
    c = size(dep_map_in,2);
    
    Y0 = round(r/2);
    X0 = round(c/2);
    
    sal_map = zeros(r,c);  
    
    if r > c
        sigma_y = sigma_x * c / r;
    else
        sigma_y = sigma_x * r / c;
    end
      
    for x = 1:c
        for y = 1:r
            sal_map(y,x) = exp(-((X0-x)^2/(2*sigma_x^2) + ((Y0-y)^2)/(2*sigma_y^2)));
        end
    end
    cbm = sal_map;    
end


