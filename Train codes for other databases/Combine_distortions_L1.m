function [FINAL_score] = Combine_distortions_L1(input)

    FINAL_score([161:240])      = input(1:80);
    FINAL_score([1:80])         = input(81:160);
    FINAL_score([81:160])       = input(161:240);
    FINAL_score([241:285])      = input(241:285);
    FINAL_score([286:365])      = input(286:365);
  

end 

