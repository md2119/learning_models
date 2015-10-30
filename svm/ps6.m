%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% November 21, 2014
% CS229
% PS6 - ps6.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objective:
%   Part a: 
%   -Plot SVM decision surface using for polynomial kernel with d=2, c=1, C=10
%   -Plot SVM decision surface using RBF kernel with sigma=1, C=10
%
%   Part b:
%   -Implement procedure to select C by cross-validation and plot
%   following:
%       -polynomial kernel sigma=0.5
%       -polynomial kernel sigma=1
%       -polynomial kernel sigma=5
%       -RBF kernel d=1 and c=1
%       -RBF kernel d=2 and c=1
%       -RBF kernel d=3 and c=1
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function [] = ps5()
    tic
    load -ascii class2d.ascii
    X=class2d(:,[1 2]);
    Y=class2d(:,3);

    part_a_RBF(X,Y,class2d,1,10);
    part_a_POLY(X,Y,class2d,2,10,1);
    
    part_b_RBF(X,Y,class2d,10,5);
    part_b_POLY(X,Y,class2d,10,5);
    clear all;
    toc
end

% part_a_RBF() plots the results for RBF Kernel in part A
function [] = part_a_RBF(X,Y,class2d,width,C)

    test_X=[-8:0.1:6];
    test_Y=[-3:0.1:11];
    [test_X,test_Y]=meshgrid(test_X,test_Y);

    K=RBF(X,X,width);
    size(K)
    pause;
    [alpha,b] = solvesvm(K, Y, C);
    [output]=test_RBF(test_X,test_Y,alpha,X,Y,width,b);
    sv=X(find(alpha~=0),:);
    str=sprintf('Part A: SVM with RBF Kernel sigma=%i C=%i',width,C);
    draw_graph(test_X,test_Y,output,class2d,sv,str);

end

% part_a_POLY() plots the results for Polynomial Kernel in part A
function [] = part_a_POLY(X,Y,class2d,d,C,c)

    test_X=[-8:0.1:6];
    test_Y=[-3:0.1:11];
    [test_X,test_Y]=meshgrid(test_X,test_Y);

    K = (X*X'+c).^d;
    [alpha,b] = solvesvm(K, Y, C);
    sv=X(find(alpha~=0),:);
    [output]=test_POLY(test_X,test_Y,X,alpha,c,b,d);
    
    str=sprintf('Part A: SVM with Polynomial Kernel d=%i C=%i c=%i',d,C,c);
    draw_graph(test_X,test_Y,output,class2d,sv,str);

end

% part_b_RBF() plots the results for RBF Kernel in part B
function []=part_b_RBF(X,Y,class2d,ns,ks)
    sigma=[0.5 1 5];
    for width=sigma
        
        avg_miss=[];
        C1=10.^(-4:0.5:4);
        for C=10.^(-4:0.5:4)
            miss=0;
            for k=1:ks    
                train_rows=randperm(size(class2d,1),size(class2d,1));
                j=1;
                for i=1:ns
                    sets(i,:)=train_rows(j:j+7);
                    j=j+8;
                end

                for n=1:ns
                    output=[];
                    train_index=setdiff(1:ns,n);
                    K=RBF(X(sets(train_index,:),:),X(sets(train_index,:),:),width);
                    [alpha,b] = solvesvm(K, Y(sets(train_index,:),:), C);
                    [output(sets(n,:),1)]=test_RBF(X(sets(n,:),1),X(sets(n,:),2),alpha,X(sets(train_index,:),:),Y(sets(n,:),:),width,b);

                    for m=sets(n,:)
                        if(or((and( (0>output(m)),(Y(m)==1) )),(and( (0<=output(m)),(Y(m)==-1) ))) )
                            miss=miss+1;
                        end
                    end

                end
            end
            avg_miss=[avg_miss miss/50];
        end
        [val,ind]=min(avg_miss);
        best_C=C1(ind);

        figure;
        test_X=[-8:0.1:6];
        test_Y=[-3:0.1:11];
        [test_X,test_Y]=meshgrid(test_X,test_Y);

        K=RBF(X,X,width);
        [alpha,b] = solvesvm(K, Y, best_C);
        [output]=test_RBF(test_X,test_Y,alpha,X,Y,width,b);
        sv=X(find(alpha~=0),:);
        str=sprintf('Gaussian Kernel SVM sigma=%i C=%i',width,best_C);
        draw_graph_2(test_X,test_Y,output,class2d,sv,str,C1,avg_miss);
        
    end
end

% part_b_POLY() plots the results for Polynomial Kernel in part B
function []=part_b_POLY(X,Y,class2d,ns,ks)
    degree=[1 2 3];
    c=1;
    for d=degree
        
        avg_miss=[];
        C1=10.^(-4:0.5:4);
        for C=10.^(-4:0.5:4)
            miss=0;
            train_rows=0;
            for kt=1:ks    
                train_rows=randperm(size(class2d,1),size(class2d,1));
                j=1;
                sets=[];
                for i=1:ns
                    sets(i,:)=train_rows(j:j+7);
                    j=j+8;
                end

                for n=1:ns
                    output=[];
                    train_index=setdiff(1:ns,n);
                    K = (X(sets(train_index,:),:)*X(sets(train_index,:),:)'+c).^d;
                    [alpha,b] = solvesvm(K, Y(sets(train_index,:),:), C);
                    [output(sets(n,:),1)]=test_POLY(X(sets(n,:),1),X(sets(n,:),2),X(sets(train_index,:),:),alpha,c,b,d);
                    
                    for m=sets(n,:)
                        if(or( (and( (0>=output(m)),(Y(m)==1) )),(and( (0<output(m)),(Y(m)==-1) ))) )
                            miss=miss+1;
                        end
                    end

                end
            end
            avg_miss=[avg_miss miss/50];
        end
        
        [val,ind]=min(avg_miss);
        
        best_C=C1(ind);
        
        figure;
        test_X=[-8:0.1:6];
        test_Y=[-3:0.1:11];
        [test_X,test_Y]=meshgrid(test_X,test_Y);

        K = (X*X'+c).^d;
        [alpha,b] = solvesvm(K, Y, best_C);
        sv=X(find(alpha~=0),:);
        [output]=test_POLY(test_X,test_Y,X,alpha,c,b,d);
    
        str=sprintf('PART B: Polynomial Kernel SVM d=%i, c=%i, C=%i',d,c,best_C);
        draw_graph_2(test_X,test_Y,output,class2d,sv,str,C1,avg_miss);
        
    end
end

% RBF() computes the RBF Kernel for given X 
function [ K ] = RBF(X,CX,width)
    dist=repmat(X,1,1,size(CX,1))-repmat(permute(CX',[3 1 2]),[size(X,1) 1 1]);    
    K=exp(-sum(dist.^2,2)./(2*width^2));
    K=permute(K,[3 1 2]);
    size(K)
    pause;
end

% test_POLY() feeds the test data to Polynomial kernel and obtain responses
function [output]=test_POLY(test_X,test_Y,CX,alpha,c,b,d)
    for c_index=1:size(CX,1)
        for row=1:size(test_X,1)
            for col=1:size(test_X,2)
                input=[test_X(row,col) test_Y(row,col)];
                Ki(row,col,c_index)=(input*CX(c_index,:)'+c).^d;
             end
        end
    end

    for w_index=1:size(alpha,1)
       fx(:,:,w_index)=alpha(w_index)*Ki(:,:,w_index);
    end  
    output=sum(fx,3)+b;
end

% test_RBF() feeds the test data to RBF kernel and obtain responses
function [output]=test_RBF(test_X,test_Y,alpha,CX,Y,width,b)
    output=[];
    
    for c_index=1:size(CX,1)
        for row=1:size(test_X,1)
            for col=1:size(test_X,2)
                input=[test_X(row,col) test_Y(row,col)];
                Ki(row,col,c_index)=exp(-sum((CX(c_index,:)-input).^2,2)./(2*width^2));
             end
        end
    end
    for w_index=1:size(alpha,1)
       fx(:,:,w_index)=alpha(w_index)*Ki(:,:,w_index);
    end
    output=sum(fx,3)+b;
end


% draw_graph() plots the results for PART A of the problem
function []=draw_graph(test_X,test_Y,output,class2d,sv,str)
    figure;
    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)==1,:)=[];
    class_1(class2d(:,3)==-1,:)=[];

    % plot contours
    [c , h] = contourf(test_X,test_Y,output,[min(min(output)) -1.0 0 1.0  max(max(output))]);
 
    %following snippet takes care f using face color of points
    map = [ 1.000 0.625 0.625 ;
        0.875 0.625 0.750 ;
        0.750 0.625 0.875 ;
        0.625 0.625 1.000 ;
      ];
    
    caxis([-2 2]);
    colormap(map);

    hold on;
    [cc,hh] = contour(test_X,test_Y,output,[0 0],'LineColor',[0 0 0],'LineWidth',3.0);    
    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
    axis equal;
    title(str);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
end

% draw_graph_2() plots the results for PART B of the problem
function []=draw_graph_2(test_X,test_Y,output,class2d,sv,str,C1,avg_miss)

    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)==1,:)=[];
    class_1(class2d(:,3)==-1,:)=[];
    
    subplot(1,2,1);
    g2=semilogx(C1,avg_miss);
  
    title('CROSS-VALIDATION CURVE');
    xlabel('C ---->');
    ylabel('C SCORE ---->');
    
    subplot(1,2,2);
    % plot contours
    [c , h] = contourf(test_X,test_Y,output,[min(min(output)) -1.0 0 1.0  max(max(output))]);
 
    %following snippet takes care f using face color of points
    map = [ 1.000 0.625 0.625 ;
        0.875 0.625 0.750 ;
        0.750 0.625 0.875 ;
        0.625 0.625 1.000 ;
      ];
    caxis([-2 2]);
    colormap(map);

    hold on;
    [cc,hh] = contour(test_X,test_Y,output,[0 0],'LineColor',[0 0 0],'LineWidth',3.0);    
    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
    axis equal;
    title(str);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
end
