%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% November 14, 2014
% CS229
% PS5 - ps5.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function calculates learned coefficients
% Input:
%   class2d.ascii
%       X - Input dataset
%       Y - Response classification
% Plots:
%   a. Classification using Radial Basis Function Network for:
%       - width 0.2, 1, 5
%       - Normalized kernel and Unnormalized kernel
%   b. Classification using K-means algorithm for:
%       - width 0.2, 1, 5
%       - Normalized kernel and Unnormalized kernel
%   c. Classification using K-Nearest Neighbour for K= 1, 3, 5, 17
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = ps5()
    load -ascii class2d.ascii
    X=class2d(:,[1 2]);
    Y=class2d(:,3);

    part_a(X,Y,class2d);
    part_b(X,Y,class2d);
    part_d(X,Y,class2d);

    clear all;
end

% part_a() plot for RBFN
function []=part_a(X,Y,class2d)
    tic
    w=[0.2 1 5];
    cnt=1;
    figure;
    for kernel=[0 1]
        for width=w 
            C=X;
            
            W=calc_kernel(X,Y,C,width,kernel);
  
            test_X=[-8:0.1:6];
            test_Y=[-3:0.1:10];
            [test_X,test_Y]=meshgrid(test_X,test_Y);
            
            [output,str1]=test_phase(test_X,test_Y,W,C,Y,width,kernel);
            
            n=2;m=3;
            str=sprintf('Gaussian Kernel %i; %s;',width, str1);
            draw_graph(test_X,test_Y,output,class2d,n,m,cnt,str);
            cnt=cnt+1;
        end
    end
    toc
end

% part_b plots for K-means
function []=part_b(X,Y,class2d)
    tic
    w=[0.2 1 5];

    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)~=0,:)=[];
    class_1(class2d(:,3)==0,:)=[];
    % select initial centroids randomly 
    for i=1:16
        seeds(i,:)=mean(X(randperm(80,5),:));
    end

    C=seeds;
    C_old=zeros(size(C));
    
    while 1
        %  Stop when centroids converge
        if(isequal(C_old,C))
            break;
        end
        C_old=C;
        
        % Find centroid which is closest to x and assign x to that cluster
        dist=repmat(X,1,1,size(C,1))-repmat(permute(C',[3 1 2]),[size(X,1) 1 1]);
        dist=sum(dist.^2,2);
        K=permute(dist,[3 1 2]);
        [z cluster]=min(K);
        
        % Recompute the centroids
        for i=1:16
            C(i,:)=mean(X(find(cluster==i),:));
            if(isnan(C(i,:)))
                C(i,:)=mean(X(randperm(80,5),:));
            end
        end
    end
    cnt=1;
    figure;
    % Find the response of test data
    for kernel=[0 1]
        for width=w 
            
            W=calc_kernel(X,Y,C,width,kernel);
            
            test_X=[-8:0.1:6];
            test_Y=[-3:0.1:10];
            [test_X,test_Y]=meshgrid(test_X,test_Y);
            
            [output,str1]=test_phase(test_X,test_Y,W,C,Y,width,kernel);
            
            n=2;m=3;
            str=sprintf('K-means %i; %s',width, str1);
            
            subplot(n,m,cnt);
            %  plot contours
            [c , h] = contourf(test_X,test_Y,output,[-Inf 0.4 0.5 0.6 Inf]);
            %
            %     %following snippet takes care f using face color of points
            colormap([0.8 0.6 0.6; 0.8 0.6 0.6;
                0.8 0.6 0.6; 0.8 0.6 0.6;
                0.6 0.6 0.8; 0.6 0.6 0.8; 0.6 0.6 0.8]);
            hold on;
            
            g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
            axis equal;
            h=plot(C(:,1),C(:,2),'blackO');
            set(h,'MarkerFaceColor',[0 0 0]);
            
            title(str,'FontSize',8);
            xlabel('X1 ---->');
            ylabel('X2 ---->');
            hold off;
            
            cnt=cnt+1;
        end
       end
    toc
end

% part_d() classifies the points using K nearest neigbours
function []=part_d(X,Y,class2d)
    tic
    Ks=[1 3 7 15];
    figure;
    for cnt=1:4
        K=Ks(cnt);

        test_X=[-8:0.1:6];
        test_Y=[-3:0.1:10];
        [test_X,test_Y]=meshgrid(test_X,test_Y);

        % For each test point:
        % a. find its distance from all training points
        % b. sort the distances
        % c. Find the deciding class based on the number of neighbouring
        % points belonging to respective class
        for row=1:size(test_X,1)
            for col=1:size(test_X,2)
                
                input=[test_X(row,col) test_Y(row,col)];
                dist_index=[sqrt(sum((X-repmat(input,[size(X,1) 1])).^2,2)) Y];
                [s, i] = sortrows(dist_index,1);
                sort_dist=dist_index(i,:);
                avg_dist=mean(sort_dist(1:K,2));
                if(mean(sort_dist(1:K,2))<0.5)
                    output(row,col)=0;
                else
                    output(row,col)=1;
                end
                
            end
        end
        n=2;m=2;
        str=sprintf('KNN K= %d',Ks(cnt));
        draw_graph(test_X,test_Y,output,class2d,n,m,cnt,str);
    end
    toc
end

% draw_graph() plots the results
function []=draw_graph(test_X,test_Y,output,class2d,n,m,cnt,str)

    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)~=0,:)=[];
    class_1(class2d(:,3)==0,:)=[];

    subplot(n,m,cnt);
    % plot contours
    [c , h] = contourf(test_X,test_Y,output,[-Inf 0.4 0.5 0.6 Inf]);
 
    % following snippet takes care f using face color of points
    colormap([0.8 0.6 0.6; 0.8 0.6 0.6;
        0.8 0.6 0.6; 0.8 0.6 0.6;
        0.6 0.6 0.8; 0.6 0.6 0.8; 0.6 0.6 0.8]);
    hold on;
    
    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    axis equal;
    title(str,'FontSize',8);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
end

% calc_kernel() compute the weights for normalized and unnormalized kernel
function [ W ] = calc_kernel(X,Y,C,width,kernel)
    dist=repmat(X,1,1,size(C,1))-repmat(permute(C',[3 1 2]),[size(X,1) 1 1]);
    dist=sum(dist.^2,2);

    K=exp(-dist./(2*width^2));
    K=permute(K,[3 1 2]);
    if(kernel==0)
        phi=K;
    else
        deno=sum(K);
        phi=K./repmat(sum(K,1),size(K,1),1);
    end
    W=(phi*phi')\(phi*Y);
    
end

% test_phase() feeds the test data to algorithm and obtain responses
function [output,str1]=test_phase(test_X,test_Y,W,C,Y,width,kernel)
    output=[];
    Kj=[];
    for c_index=1:size(C,1)
        for row=1:size(test_X,1)
            for col=1:size(test_X,2)
                input=[test_X(row,col) test_Y(row,col)];
                Ki(row,col,c_index)=exp(-sum((C(c_index,:)-input).^2,2)./(2*width^2));
            end
        end
    end
    Kj=sum(Ki,3);

    if(kernel==0)
        str1='Unnormalized';
        for w_index=1:size(W,1)
            fx(:,:,w_index)=W(w_index)*Ki(:,:,w_index);
        end
    else
        str1='Normalized';
        for w_index=1:size(W,1)
            fx(:,:,w_index)=W(w_index)*(Ki(:,:,w_index)./Kj);
        end
    end
    output=sum(fx,3);
end
