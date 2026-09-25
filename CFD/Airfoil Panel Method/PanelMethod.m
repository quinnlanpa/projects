% Quinn Aiken
% Panel method applied to naca 0012 
clc
clear 
close all

% plot panel nodes
% num points in number of panel vortex on half of the airfoil-1.
% i.e. if you want 50 points choose numpoints=26
numpoints=51;
% numpoints=6;
% % use halfcircle to get more points at LE and TE
theta=linspace(0,pi,numpoints);
xpoints=0.5*cos(theta)+0.5;
ypoints=naca(xpoints);
% mirror for bottom of circle
panelpoints=[xpoints flip(xpoints(2:end-1))
    ypoints flip(-ypoints(2:end-1))];
panelpoints=panelpoints';


% panels is a matrix which in each of its rows holds index (row that contains
% coords in panelpoints)of boundary points for each panel 
% below is code verfying it describes all panels in correct order
panelindex1=linspace(1,size(panelpoints,1)-1,size(panelpoints,1)-1);
panelindex2=panelindex1+1;
panels=[panelindex1' panelindex2';
    size(panelpoints,1) 1];

for i=1:size(panels,1)
    plot(panelpoints(panels(i,:),1),panelpoints(panels(i,:),2),'o-')
    hold on
    % pause(.25)
end
title("Naca 0012 airfoil shape")
xlabel('x/c')
ylabel('y')
axis('equal')

% form vector lengths of each panel
lengthS=zeros(size(panels,1),1);
% form vector of control points
% row 1 is cords for cp for panel 1
cp=zeros(size(panels,1),2);
% form vector of angles 
ptheta=zeros(size(panels,1),1);
for i=1:size(panels,1)
    BoundaryPoint1=panelpoints(panels(i,1),:);
    BoundaryPoint2=panelpoints(panels(i,2),:);
    lengthS(i)=norm(BoundaryPoint2-BoundaryPoint1,2);
    
    % for cp
    cp(i,1)=0.5*(BoundaryPoint2(1)+BoundaryPoint1(1));
    cp(i,2)=0.5*(BoundaryPoint2(2)+BoundaryPoint1(2));

    % for theta
    ptheta(i)=atan2(BoundaryPoint2(2)-BoundaryPoint1(2),BoundaryPoint2(1)-BoundaryPoint1(1));
end

% for i=1:

% add control points to end of panelpoints list
% each row hold the coords for a given boundary or control point
panelpointsc=[panelpoints;cp];
% add there index to panels in 3rd column
panelsc=zeros(size(panelpoints,1),3);
panelsc(:,1:2)=panels;
temp=linspace(1,size(cp,1),size(cp,1));
temp=temp+size(panelpoints,1);
panelsc(:,3)=temp';
% the jth row describes the index in panelpointsc of the boundary points
% in the first two columns and the index of the conrol point in the third 
% column for the jth panel

% verify panelsc is working prperly
% figure
% % keyboard
% for i=1:size(panelsc,1)
%     plot(panelpointsc(panelsc(i,1:2),1),panelpointsc(panelsc(i,1:2),2),'o-')
%     hold on
%     plot(panelpointsc(panelsc(i,3),1),panelpointsc(panelsc(i,3),2),'*')
%     % pause(.25)
% end
% title('A')

% verify cp is working properly
% figure 
% for i=1:size(panelpoints,1)
%     plot(panelpoints(i,1),panelpoints(i,2),'o-')
%     hold on
%     plot(cp(i,1),cp(i,2),'*')
%     % pausec
% end

m=size(panelpoints,1);

% compute cn1,cn2,ct1,ct2
Cn1=zeros(m,m);
Cn2=zeros(m,m);
Ct1=zeros(m,m);
Ct2=zeros(m,m);

for i=1:m
    for j=1:m
        xic=panelpointsc(panelsc(i,3),1);
        yic=panelpointsc(panelsc(i,3),2);
        xjb=panelpointsc(panelsc(j,1),1);
        yjb=panelpointsc(panelsc(j,1),2);

        A=-(xic-xjb)*cos(ptheta(j))-(yic-yjb)*sin(ptheta(j));
        B=(xic-xjb)^2+(yic-yjb)^2;
        C=sin(ptheta(i)-ptheta(j));
        D=cos(ptheta(i)-ptheta(j));
        E=(xic-xjb)*sin(ptheta(j))-(yic-yjb)*cos(ptheta(j));
        F=log(1+(lengthS(j)^2+2*A*lengthS(j))/(B));
        G=atan2(E*lengthS(j),B+A*lengthS(j));
        P=(xic-xjb)*sin(ptheta(i)-2*ptheta(j))+(yic-yjb)*cos(ptheta(i)-2*ptheta(j));
        Q=(xic-xjb)*cos(ptheta(i)-2*ptheta(j))-(yic-yjb)*sin(ptheta(i)-2*ptheta(j));
        
        Cn2(i,j)=D+(Q*F)/(2*lengthS(j))-(G*(A*C+D*E))/(lengthS(j));
        Cn1(i,j)=0.5*D*F+C*G-Cn2(i,j);

        Ct2(i,j)=C+(P*F)/(2*lengthS(j))+(G*(A*D-C*E))/(lengthS(j));
        Ct1(i,j)=0.5*C*F-D*G-Ct2(i,j);
    end
end
% hardcode diagonal values to eliminate rounding error
for i=1:m
    Cn1(i,i)=-1;
    Cn2(i,i)=1;
    Ct1(i,i)=-pi/2;
    Ct2(i,i)=-pi/2;
end

% check structure of matrix through visualisation
% keyboard
% figure
% imagesc(Cn1)
% title('Cn1')
% figure
% imagesc(Cn2)
% title('Cn2')

AA=zeros(m+1,m+1);
b=zeros(m+1,1);

% form An called AA here
AA(1:m,1)=Cn1(1:m,1);
for i=1:m
    for j=2:m
        AA(i,j)=Cn1(i,j)+Cn2(i,j-1);
    end
end
AA(1:m,m+1)=Cn1(1:m,m);
AA(m+1,:)=0;
AA(m+1,1)=1;
AA(m+1,m+1)=1;
% visualize
% figure
% imagesc(AA)
% title('An')
% keyboard
% --------------------------------------------------
% choose alpha
alpha=6;
alpha=deg2rad(alpha);

% form b
b(1:m)=sin(ptheta-alpha);
b(m+1)=0;

% solve system
gamma=AA\b;

% form At
At=zeros(m,m+1);
At(:,1)=Ct1(:,1);
for i=1:m
    for j=2:m
        At(i,j)=Ct1(i,j)+Ct2(i,j-1);
    end
end
At(:,m+1)=Ct2(:,m);

% solve for U
U=cos(ptheta-alpha)+At*gamma;
% compute pressure
pressurefromPanel=1-U.^2;

figure
plot(cp(numpoints+1:end,1),pressurefromPanel(numpoints+1:end,1))
hold on
plot(cp(1:numpoints,1),pressurefromPanel(1:numpoints,1))
ylim([-3 1])
set(gca,'YDir','reverse')
title('C_p from panel mathod vs experimental for alpha = 6 degrees')
xlabel('x/c')
ylabel('C_p')
hold on

% plot value of true Cp from experimental data

AoA5lower=readmatrix("cp2\AoA6lower.csv");
plot(AoA5lower(:,1),AoA5lower(:,2),'o')
hold on 
AoA5upper=readmatrix("cp2\AoA6upper.csv"); 
plot(AoA5upper(:,1),AoA5upper(:,2),'*')
legend('PM lower','PM upper', 'EX lower', 'EX upper')

% run again for AoA=10
% choose alpha
alpha=10;
alpha=deg2rad(alpha);

% form b
b(1:m)=sin(ptheta-alpha);
b(m+1)=0;

% solve system
gamma=AA\b;
U=cos(ptheta-alpha)+At*gamma;
% U=U/norm(U);
% compute pressure
pressurefromPanel=1-U.^2;

figure
plot(cp(numpoints+1:end,1),pressurefromPanel(numpoints+1:end,1))
hold on
plot(cp(1:numpoints,1),pressurefromPanel(1:numpoints,1))
ylim([-6 1])
set(gca,'YDir','reverse')
% legend('upper','lower')
title('C_p from panel mathod vs experimental for alpha = 10 degrees')
xlabel('x/c')
ylabel('C_p')
hold on

% plot value of true Cp from experimental data
AoA5lower=readmatrix("cp2\AoA10lower.csv"); 
plot(AoA5lower(:,1),AoA5lower(:,2),'o')
hold on 
AoA5upper=readmatrix("cp2\AoA10upper.csv"); 
plot(AoA5upper(:,1),AoA5upper(:,2),'*')
legend('PM lower','PM upper', 'EX lower', 'EX upper')

function y=naca(x)
    y= 0.594689181*(0.298222773*sqrt(x) - 0.127125232*x - 0.357907906*x.^2 + 0.291984971*x.^3 - 0.105174606*x.^4);
end