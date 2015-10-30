%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% November 7, 2014
% CS229
% PS4 - ps4.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function calculates learned coefficients
% Input:
%   class2d.ascii
%       X - Input dataset
%       Y - Response class
% Output:
%       w - Neural Network Learned weights
% Plot:
%       nonlinear decision boundary learned from Neural Network
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = ps4 ()
    lambda=[0.001 0.01 0.1];
    hunits=[20 5 1];
    
    load -ascii class2d.ascii
  
    % adding constant one to each input vactor (bias)
    X=[ones(size(class2d,1),1) class2d(:,[1 2])];
    Y=class2d(:,3);

    
    for decay=lambda
        for units=hunits
            
            W2=2.*rand(units+1,1)-1;
            W1=2.*rand(3,units)-1;
            if(units==1)
                stop_err=0.15;
            elseif(units==5)
                stop_err=0.1;
            elseif(units==20)
                stop_err=0.1;
            end
            
            hidden_1(decay,units,W1,W2,X,Y,class2d,stop_err);
            
        end
    end

    for decay=lambda
        for units=hunits
       
            if(units==1)
                stop_err=0.15;
            elseif(units==5)
                stop_err=0.1;
            elseif(units==20)
                stop_err=0.1;
            end
            
            W3=2.*rand(units+1,1)-1;
            W2=2.*rand(units+1,units)-1;
            W1=2.*rand(3,units)-1;

            hidden_2(decay,units,W1,W2,W3,X,Y,class2d,stop_err);
        end
    end
end


function []= hidden_1(decay,units,W1,W2,X,Y,class2d,stop_err)
    tic
    str=sprintf('Layers: 2 Units:%f Decay:%f',units,decay);
    disp(str);
    i=1;
    loss_arr=[];
    while 1
        eta=1/(11+i);
        i=i+1;
        Z1=[ones(80,1) exp(X*W1)./(1+exp(X*W1))];
        Z2=exp(Z1*W2)./(1+exp(Z1*W2));
        
        loss=-(Y.*log10(Z2)+(ones(80,1)-Y).*log10(ones(80,1)-Z2));
        mean(loss)
        if(mean(loss)<stop_err);
            break;
        end
        
        DELTA2=Z2-Y;
        D=Z1(:,2:end).*(ones(80,units)-Z1(:,2:end));
        DELTA1=D.*(W2(2:end,:)*DELTA2')';
    
        update2=-eta.*(Z1'*DELTA2)-eta*decay*W2;
        update1=-eta.*(X'*DELTA1)-eta*decay*W1;

        W2=W2+update2;
        W1=W1+update1;

    end
    

    X=[-8:1:6];
    Y=[-3:1:10];
    [X,Y]=meshgrid(X,Y);

    output=[];
    for row=1:size(X,1)
        for col=1:size(X,2)
            input=[1 X(row,col) Y(row,col)];
            T1=[1 exp(input*W1)./(1+exp(input*W1))];

            T2=exp(T1*W2)./(1+exp(T1*W2));

            output(row,col)=T2;
        end
    end
    
    figure;
    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)~=0,:)=[];
    class_1(class2d(:,3)==0,:)=[];
    
    % plot contours 
    [c , h] = contourf(X,Y,output,[-Inf 0.4 0.5 0.6 Inf]);
    
    %following snippet takes care f using face color of points
    colormap([0.8 0.6 0.6; 0.8 0.6 0.6;
        0.8 0.6 0.6; 0.8 0.6 0.6;
        0.6 0.6 0.8; 0.6 0.6 0.8; 0.6 0.6 0.8]);
    hold on;

    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    title(str);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
disp('PRESS ANY KEY FOR NEXT PLOT...');
pause;
    %plot_graph(X,Y,output,class2d,str);
    
    toc
    return;
end


function []= hidden_2(decay,units,W1,W2,W3,X,Y,class2d,stop_err)
    tic
    str=sprintf('Layers: 3 Units:%f Decay:%f',units,decay);
    disp(str);
    i=1;
    loss_arr=[];
    while 1
        eta=10/(50+i);
        i=i+10;
        Z1=[ones(80,1) exp(X*W1)./(1+exp(X*W1))];
        Z2=[ones(80,1) exp(Z1*W2)./(1+exp(Z1*W2))];
        Z3=exp(Z2*W3)./(1+exp(Z2*W3));
        
        loss=-(Y.*log10(Z3)+(ones(80,1)-Y).*log10(ones(80,1)-Z3));
        mean(loss)
        if(mean(loss)<stop_err);
            break;
        end
        
        DELTA3=Z3-Y;
        D=Z2(:,2:end).*(ones(80,units)-Z2(:,2:end));
        DELTA2=D.*(W3(2:end,:)*DELTA3')';
        
        D=Z1(:,2:end).*(ones(80,units)-Z1(:,2:end));
        DELTA1=D.*(W2(2:end,:)*DELTA2')';

        update3=-eta.*(Z3'*DELTA3)-eta*decay.*W3;
        update2=-eta.*(Z2'*DELTA2)-eta*decay.*W2;
        update1=-eta.*(X'*DELTA1)-eta*decay.*W1;

        W3=W3+update3;
        W2=W2+update2;
        W1=W1+update1;

    end

    
    X=[-8:1:6];
    Y=[-3:1:10];
    [X,Y]=meshgrid(X,Y);

    output=[];
    for row=1:size(X,1)
        for col=1:size(X,2)
            input=[1 X(row,col) Y(row,col)];
            T1=[1 exp(input*W1)./(1+exp(input*W1))];

            T2=[1 exp(T1*W2)./(1+exp(T1*W2))];

            T3=exp(T2*W3)./(1+exp(T2*W3));

            output(row,col)=T3;
        end
    end

    
    
    figure;
    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)~=0,:)=[];
    class_1(class2d(:,3)==0,:)=[];
    
    % plot contours 
    [c , h] = contourf(X,Y,output,[-Inf 0.4 0.5 0.6 Inf]);
    
    %following snippet takes care f using face color of points
    colormap([0.8 0.6 0.6; 0.8 0.6 0.6;
        0.8 0.6 0.6; 0.8 0.6 0.6;
        0.6 0.6 0.8; 0.6 0.6 0.8; 0.6 0.6 0.8]);
    hold on;

    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    title(str);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
pause;

    toc
    return;
end


function []=plot_graph(X,Y,output,class2d,str)
    figure;
    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)~=0,:)=[];
    class_1(class2d(:,3)==0,:)=[];
    
    % plot contours 
    [c , h] = contourf(X,Y,output,[-Inf 0.4 0.5 0.6 Inf]);
    
    %following snippet takes care f using face color of points
    colormap([0.8 0.6 0.6; 0.8 0.6 0.6;
        0.8 0.6 0.6; 0.8 0.6 0.6;
        0.6 0.6 0.8; 0.6 0.6 0.8; 0.6 0.6 0.8]);
    hold on;

    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    title(str);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
pause;
end
