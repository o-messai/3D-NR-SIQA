function  [rocc,Krooc, cc,rmse]= logistic_distortions_cc_L1 (x,y)
% [cc,rocc, mae, rmse,Krooc]
%[cc, rocc, mae, rmse, or,Krooc ,quality]
%x=les valeur des methodes
%y est DMOS
temp=corrcoef(x,y);

if (temp(1,2)>0)
    t(3) = mean(x);
    t(1) = (abs(max(y) - min(y)));
    t(4)=   mean(y);
    t(2) =  (1/std(x));
    t(5)=-1;
    signslope=1;
else
    t(3) = mean(x);
    t(1) = -(abs(max(y) - min(y)));
    t(4)=  mean(y);
    t(2) = (1/std(x));
    t(5)=-1;
    signslope=-1;    
end


 tt=fminsearch(@errfun, t, optimset('Display','off'), x,y, signslope);
% cc = corrcoef(fitfun(tt,x),y);
% rmse = sqrt(mean((fitfun(tt,x)-y).^2));
% plot(x,y,'x'), hold on, plot (sort(x), fitfun(tt,sort(x))), hold off
% 
% function e = errfun(t,x,y)
% e = sum((y-fitfun(t,x)).^2);
% 
% function f = fitfun(t,x)
% f=t(1).*(logistic(t(2), (x-t(3))))+t(4)+t(5).*x;
% 
% function f = logistic(t,x)
% f = 0.5-(1./(1+exp(t.*x)));

% tt=fminunc(@errfun, t, optimset('Display','off', 'LargeScale','off'), x,y, signslope);
%tt=fminunc(@errfun, t, x,y, signslope);
%tt=fminsearch(@errfun, t, optimset('Display','off'), x,y, signslope);

s = size (y);
%yfit = fitfun(tt,x,signslope);(s(1).^2-1))
cc1 = corrcoef(fitfun(tt,x, signslope),y);


%beta(1) = max(y);
%beta(2) = 10;
%beta(3) = mean(x);
%beta(4) = 0.1;
%beta(5) = 0.1;
%betatt=fminunc(@errfun, beta, optimset('Display','off', 'LargeScale','off'), x,y, signslope);
%corr_coef = corr(y, fitfun(betatt,x, signslope), 'type','Pearson'); %pearson linear coefficient
%corr_coef = corrcoef(fitfun(betatt,x, signslope),y) ; %pearson linear coefficient
cc=cc1(1,2);
rmse = sqrt(mean((fitfun(tt,x, signslope)-y).^2));

%%%%%%%%%%pour tous les mesure sauf PSNR
%rocc=1-(6* mean((fitfun(tt,x, signslope)-y).^2) /(s(1).^2-1));  

%%%%%%%%%%pour PSNR toute la base
%rocc=1-(600*1.55* mean((fitfun(tt,x, signslope)-y).^2) /(s(1).^2-1));

%%%%%%%%%%pour MSSIM 
%rocc=1-(600* mean((fitfun(tt,x, signslope)-y).^2) /(s(1).^2-1));
%rocc=100*(rocc-0.99);
%rocc=1-(6* sum(sum(((fitfun(tt,x, signslope)-y).^2))) /(s(1).^3-s(1)));

rocc=spear(fitfun(tt,x, signslope),y);

% Krooc=kendall(fitfun(tt,x, signslope),y);

Krooc= corr(y,fitfun(tt,x, signslope),'type', 'kendall');



mae=mean(abs(fitfun(tt,x, signslope)-y)); 

%s=size(z);
compte=0;
for i=1:s(1),
    
 %if (abs(fitfun(tt,x(i), signslope)-y(i))> 2*z(i))
 compte=compte+1;
 %end
end;

%or=10*compte/s(1);
or=10*compte/s(1);

wn_x = x([161:240]) ;
wn_y = y([161:240]) ;
%wn_z = PP_L1([161:240]) ;

jp2_x = x([1:80]) ; 
jp2_y = y([1:80]) ; 
%jp2_z = PP_L1([1:80]) ; 

jpeg_x = x([81:160]) ;    
jpeg_y = y([81:160]) ; 
%jpeg_z = PP_L1([81:160]) ; 

gb_x = x([241:285]) ;   
gb_y = y([241:285]) ;     
%gb_z = PP_L1([241:285]) ;    

ff_x = x([286:365]) ;   
ff_y = y([286:365]) ;  
%ff_z = PP_L1([286:365]) ;  

set(gca,'FontSize',11,'FontWeight','bold')
plot(wn_x,wn_y,'p',jp2_x,jp2_y,'o',jpeg_x,jpeg_y,'<',gb_x,gb_y,'*',ff_x,ff_y,'s'), hold on, plot (sort(x), fitfun(tt,sort(x), signslope),'r','LineWidth',1.7), hold off
%legend('JPEG2000 Images (JP2K_1)','Fitting with logistic function');
%legend('JPEG2000 Images ','Fitting with logistic function');
%legend('JPEG2000 Images (JP2K_2)','Fitting with logistic function');
%legend('JPEG2000 Images','Fitting with logistic function');
%legend('JPEG Images(JPEG_1)','Fitting with logistic function');
%legend('JPEG Images(JPEG_2)','Fitting with logistic function');
%legend('White noise Images','Fitting with logistic function');
%legend('Gaussian blur Images','Fitting with logistic function');
%legend('fast-fading Rayleigh Images','Fitting with logistic function');
%legend('All Images','Fitting with logistic function');
%set(h,'Interpreter','none');h =
%legend('toyama','Fitting with logistic function');
legend('WN','JP2K','JPEG','Gblur','FF','Logistic function','FontSize',10,'FontWeight','bold');

%xlabel('ERTDM');
%xlabel('ERTDM 15*15');
%xlabel('ERTDM 30*30');
%xlabel('ERTDM 8*8');
%xlabel('ERTDM 11*11');
%xlabel('BNBM');
%xlabel('ERDDM new');
%xlabel('jp2knr');
xlabel('Predicted Quality score (LIVE I)','FontSize',11,'FontWeight','bold');
%xlabel('ERDDM 15*15');
%xlabel('ERDDM 8*8');
%xlabel('ERDDM 11*11');


%xlabel('WroiWQI');
%xlabel('GSSIM');
%xlabel('MSSIM');
%xlabel('PSNR');
ylabel('DMOS','FontSize',11,'FontWeight','bold');
%ylabel('MOS');

%title(['Temperature is ','C']);

function e = errfun(t,x,y,signslope)
e = sum((y-fitfun(t,x, signslope)).^2);
%pour calculer DMOSp
function f = fitfun(t,x,signslope)
f=(t(1)).*(logistic(t(2), (x-t(3))))+t(4)+signslope.*exp(t(5)).*x;

function f = logistic(t,x)
f = 0.5-(1./(1+exp((t).*x)));


function r=spear(x,y)

% x and y must have equal number of rows
if size(x,1)~=size(y,1)
    error('x and y must have equal number of rows.');
end


% Find the data length
N = length(x);

% Get the ranks of x
R = crank(x)';

for i=1:size(y,2)
    
    % Get the ranks of y
    S = crank(y(:,i))';
    
    % Calculate the correlation coefficient
    r(i) = 1-6*sum((R-S).^2)/N/(N^2-1);
    
end

% Calculate the t statistic
if r == 1 | r == -1
    t = r*inf;
else
    t=r.*sqrt((N-2)./(1-r.^2));
end

% Calculate the p-values
p=2*(1-tcdf(abs(t),N-2));





function r=crank(x)
u = unique(x);
[xs,z1] = sort(x);
[z1,z2] = sort(z1);
r = (1:length(x))';
r=r(z2);

for i=1:length(u)
    
    s=find(u(i)==x);
    
    r(s,1) = mean(r(s));
    
end