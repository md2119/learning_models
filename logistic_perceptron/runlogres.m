%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% October 31, 2014
% CS229
% PS3 - runlogres.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function calculates learned coefficients
% Input:
%   class2d.ascii
%       X - Input dataset
%       Y - Response classification
% Output:
%       Learned weights
% Plot:
%       Separating hyperplane using logistic regression
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = runlogres ()
    tic
    load -ascii class2d.ascii
    class_1=class2d;
    class_0=class2d;
 
    % extracting class -1 and class 1 from input array class2d
    class_0(class2d(:,3)>0,:)=[];
    class_1(class2d(:,3)<0,:)=[];
    Y=class2d(:,3);
  
    % adding constant one to each input vactor (bias)
    X=[ones(size(class2d,1),1) class2d(:,[1 2])];
    beta_new=zeros(3,1);
    
    figure;
    i=1;
    while 1 
        disp('iteration=======>');
        disp(i);
        i=i+1;
        
        % compute the fitted probability for each element 
        p=sigmoid(beta_new, X, Y);
        
        % compute the NxN Hessian matrix
        W=diag(p.*(ones(size(p))-p));
        
        beta_old=beta_new;
        
        % Calcuate adjusted response
        z=X*beta_new+(Y./p);
        
        % Newton's step
        beta_new=inv(X'*W*X)*X'*W*z;
    
        % convergence criteria for logistic, if learned coefficients are
        % not varying not too, stop.
        if(abs(beta_new-beta_old)<=0.0001)
            break;
        end
       
        % plot the decision boundary after each iteration i.e for each new
        % learnt weight
        plot1=plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');
        db_line=drawline(beta_old(2:end),beta_old(1));
        db_line.Color='black';
        db_line.LineWidth=2;
        
        title('PLOT 1: LOGISTIC REGRESSION');
        xlabel('X1 ---->');
        ylabel('X2 ---->');
        legend('y=-1','y=1');
        pause(2);
    end
    toc
end

% this function computes the fitted probabilities for all elements
function [p]=sigmoid (beta, X, Y)
    p=exp(Y.*(X*beta))./(1+exp(Y.*(X*beta)));
end