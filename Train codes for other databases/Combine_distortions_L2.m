function [FINAL_score] = Combine_distortions_L2(input)

 

    FINAL_score([1:9 46:54 91:99 136:144 181:189 226:234 271:279 316:324])      = input(1:72);
    FINAL_score([10:18 55:63 100:108 145:153 190:198 235:243 280:288 325:333])  = input(73:144);
    FINAL_score([19:27 64:72 109:117 154:162 199:207 244:252 289:297 334:342])  = input(145:216);
    FINAL_score([28:36 73:81 118:126 163:171 208:216 253:261 298:306 343:351])  = input(217:288);
    FINAL_score([37:45 82:90 127:135 172:180 217:225 262:270 307:315 352:360])  = input(289:360);
  

end 

