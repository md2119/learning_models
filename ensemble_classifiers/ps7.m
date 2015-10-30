%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% December 5, 2014
% CS229
% PS7 - ps7.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objective:
%   Part a: 
%   -Plot decision surface using Bagging with varying number of trees
%
%   Part b: 
%   -Plot decision surface by refitting bagged coeficients with logistic
%   with varying number of trees
%
%   Part c: 
%   -Plot decision surface using boosting with varying number of trees
%   
%   Part d: 
%   -Plot testing error on spamtest.ascii using above methods
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function []=ps7()
    load -ascii class2d.ascii
    depth=3;leaf=5;
    figure;
    tree_num = [10 20 50 100];
    t = part_a(class2d, tree_num, depth, leaf);

    figure;
    part_b(class2d, t, 100, depth, leaf);

    figure;
    part_c(class2d, tree_num, depth, leaf);

    depth=6;leaf=100;tree_num = [ 20:20:200];
    load -ascii spamtrain.ascii;
    load -ascii spamtest.ascii;
    part_d(spamtrain, spamtest, depth, leaf, tree_num);
        
end

function [t] = part_a(class2d, tree_num, depth, leaf)
% part_s plots the decision surface using bagging

    tmax = max(tree_num);
    
    for i=1:tmax
        rows=ceil(80*rand(size(class2d,1),1));
        Y=class2d(rows,3);
        X=class2d(rows,[1 2]);
        t{i}=traindt(X, Y, depth, @splitentropy, leaf); 
    end
    
    test_X=[-8:0.1:6];
    test_Y=[-3:0.1:11];
    [test_X,test_Y]=meshgrid(test_X,test_Y);
    [nx,ny] = size(test_X);
    
    input = [reshape(test_X',nx*ny,1) reshape(test_Y',nx*ny,1)];
    for cnt=1:1:size(tree_num,2)
        for t_ind=1:1:tree_num(cnt)
            output(:,t_ind) = dt(input,t{t_ind});
        end
        Z=(reshape(sum(output,2), nx, ny))'./tree_num(cnt);
        str=sprintf('Bagging d=%i',tree_num(cnt));
        draw_graph_2(test_X,test_Y,Z,class2d,str, cnt);
        cnt=cnt+1;
    
    end
    
end

function [] = part_b(class2d, t, tree_num, depth, leaf)
% part_b plots decision surface by refitting the coeficients obtained for
% bagging with 200 trees

    cnt=1;
    for lambda=[0 2 5 10]
        Y = class2d(:,3);
        X = class2d(:,[1 2]);
        for t_ind = 1:tree_num
            Xb(:,t_ind) = dt(X,t{t_ind});
        end
        w = trainlogregL1(Xb, Y, lambda);
        test_X=[-8:0.1:6];
        test_Y=[-3:0.1:11];
        [test_X, test_Y]=meshgrid(test_X, test_Y);
        [nx,ny] = size(test_X);

        input = [reshape(test_X',nx*ny,1) reshape(test_Y',nx*ny,1) ones(nx*ny,1)];

        for t_ind=1:tree_num
            output(:,t_ind) = w(t_ind)*dt(input,t{t_ind});
        end
        
        ntree = nnz(w);
        
        Z=(reshape(sum(output,2), nx, ny))';
        str=sprintf('Bagging with Logistic lambda=%d d=%d',lambda, ntree);
        draw_graph_2(test_X,test_Y,Z,class2d,str,cnt);
        cnt=cnt+1;
    end
end

function [] = part_c(class2d, tree_num, depth, leaf)
    
    tmax = max(tree_num);
    Y = class2d(:,3);
    X = class2d(:,[1 2]);
        
    alpha = ones (size(class2d,1),1);
    [w, t] = train_boost(X, Y, alpha, tmax, depth, leaf);
    
    test_X=[-8:0.1:6];
    test_Y=[-3:0.1:11];
    [test_X, test_Y]=meshgrid(test_X, test_Y);
    
    for cnt=1:1:size(tree_num,2)
        Z = test_boost(test_X, test_Y, t(1:tree_num(cnt)), tree_num(cnt), w(1:tree_num(cnt)));
        str=sprintf('Adaboost d=%i',tree_num(cnt));
        draw_graph_2(test_X,test_Y,Z,class2d,str,cnt);
    end
    
end

function [] = part_d(traininp, testinp, depth, leaf, tree_num)
    
%%%% error rate on spamtest using bagging

    tmax = max(tree_num);
    
    for i=1:tmax
        rows=ceil(size(traininp,1)*rand(size(traininp,1),1));
        t{i}=traindt(traininp(rows,1:57), traininp(rows,58), depth, @splitentropy, leaf); 
    end
    err=[];
    for cnt=1:1:size(tree_num,2)
        output=[];Z=[];miss=[];
        for t_ind=1:1:tree_num(cnt)
            output(:,t_ind) = dt(testinp(:,1:57),t{t_ind});
        end
        Z=mean(output,2);
        
        miss = calc_miss(Z,testinp(:,58) );
        err=[err mean(miss)];
    end
    err
    g1=plot(20:20:200,err,'LineWidth',2);
    hold on;

    %%%%% error rate on spamtest using bagging with logistic regression
    for t_ind = 1:200
        Xb(:,t_ind) = dt(traininp(:,1:57),t{t_ind});
    end

    for t_ind=1:200
        intop(:,t_ind) = dt(testinp(:,1:57),t{t_ind});
    end
    ntree=[];err=[];
    for lambda=[0:0.5:5]
        w = trainlogregL1(Xb, traininp(:,58), lambda);
        
        for t_ind=1:1:max(tree_num)
            output(:,t_ind) = w(t_ind)*intop(:,t_ind);
        end
        Z = mean(output,2);
        miss = calc_miss(Z,testinp(:,58) );
        err = [err mean(miss)];
        ntree = [ ntree nnz(w)];
        
    end
    plot(ntree,err,'LineWidth',2);
        
     %%%% error rate on spamtest using boosting
     err=[];
     tmax = max(tree_num);
     alpha = ones (size(traininp,1),1);
     [w, t] = train_boost(traininp(:,1:57), traininp(:,58), alpha, tmax, depth, leaf);
 
     for cnt=1:1:size(tree_num,2)
         output=[];miss=[];Z=[];
         for t_ind=1:tree_num(cnt)
             output(:,t_ind) = w(t_ind) *dt(testinp(:,1:57),t{t_ind});
         end
         Z = sum(output,2);
         miss = calc_miss(Z,testinp(:,58) );
         err=[err mean(miss)];
     end
     err
    
    str=sprintf('Test Error on SPAM Data');
    g1=plot(20:20:200,err);
    title(str);
    xlabel('Number of trees ---->');
    ylabel('Error rate ---->');
    legend('Bagging','Bagging+logistic','Boosting');
    hold off;
    
end

% draw_graph_2() plots the results for PART B of the problem
function []=draw_graph_2(test_X,test_Y,output,class2d,str,i)
    class_1=class2d;
    class_0=class2d;

    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)==1,:)=[];
    class_1(class2d(:,3)==-1,:)=[];
    
    subplot(2,2,i);
    % plot contours
    [c , h] = contourf(test_X,test_Y,output,[min(min(output)) 0 max(max(output))]);
 
    %following snippet takes care f using face color of points
    map = [ 1.000 0.625 0.625 ;
        0.625 0.625 1.000 ;
      ];
    caxis([-2 2]);
    colormap(map);

    hold on;
    g1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
    
    axis equal;
    title(str);
    xlabel('X1 ---->');
    ylabel('X2 ---->');
    hold off;
end

function [w, t] = train_boost(X, Y, alpha, tree_num, depth, leaf)
% train_boost() computes coefficients using Adaboost by training on given
% data

    t=cell(tree_num,1);
    w=[];
    for i=1:tree_num
        miss=[];
        t{i} = traindtw(X, Y, alpha, depth, @splitentropy, leaf);
        fm = dt(X,t{i});
        miss = calc_miss(fm, Y);
    
        err = (alpha'*miss)/sum(alpha);
        w(i) = log((1-err)/err);
        alpha=alpha .* exp(w(i).*miss);
        
    end
end

function [Z] = test_boost (test_X, test_Y, t, tree_num, w)
% test_boost() computes the decision boundary on test points using the
% coeficients from Adaboost
    input=[];
    [nx,ny] = size(test_X);
    input = [reshape(test_X',nx*ny,1) reshape(test_Y',nx*ny,1)];
    for t_ind=1:tree_num
        output(:,t_ind) = w(t_ind) *dt(input,t{t_ind});
    end
    Z=(reshape(sum(output,2), nx, ny))';

end

function [miss] = calc_miss(Z,Y)
% calc_miss() calcuates the number of missclassified points

    miss = sign(Z.*Y);
    miss(miss>0) = 0;
    miss(miss<0) = 1;
end