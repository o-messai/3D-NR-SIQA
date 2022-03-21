
function [final_smap] = norm_operation(smap)

    
% smap = mat2gray(smap);
[row col] = size(smap);
final_smap = mat2gray(smap);
[salient_row salient_col salient_value] = find(final_smap > 0.9);

%% csf for enhance
salient_distance = zeros(row, col);
salient_csf = zeros(row, col);
for i = 1:row
    for j = 1:col        
        salient_distance(i, j) = min(sqrt((i - salient_row).^2 + (j - salient_col).^2));        
    end
end
salient_distance = mat2gray(salient_distance);

salient_csf(:, :) = 64./(exp(256 * 0.106 * pi * (atan(8 * salient_distance./ 1536) + 2.3) / (2.3 * 180)));

salient_csf = mat2gray(salient_csf);
final_smap = final_smap .*  salient_csf;

final_smap = mat2gray(final_smap);

end