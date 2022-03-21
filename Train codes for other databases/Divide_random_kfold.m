function [trainIdxs,testIdxs,train_IN,train_OUT,test_IN] = Divide_random_kfold(input,output,k)


    
    cv = cvpartition(length(output), 'kfold',k);
    
    for i=1:k
        
        trainIdxs{i} = find(training(cv,i)); 
        testIdxs{i}  = find(test(cv,i));     
        
        
        train_IN{i} = [input(:,:,:,trainIdxs{i})]; %Inout and output for the network 80%
        train_OUT{i} = [output(trainIdxs{i})];
        
        test_IN{i} = [input(:,:,:,testIdxs{i})];  %Input for Testing 20%
        %test_OUT{i} = [output(testIdxs{i})];
      
    end
    
    
end

