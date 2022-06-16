    function [output] = concatenate_data(input,S)
    output = []; 
    for i=1:length(input)
        output = cat(S,output, input{i});
    end
    end

