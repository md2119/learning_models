%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% October 31, 2014
% CS229
% PS3 - runpercep.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function calculates learned coefficients
% Input:
%   class2d.ascii
%       X - Input dataset
%       Y - Response class
% Output:
%       w - Learned weights
% Plot:
%       separating hyperplane using perceptron learning algorithm
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = runpercep ()
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
    
    % initially coefficients/weights are assigned to be zero
    w=zeros(1,3);
    
    % start with learning rate 0.1 and reduce to 80% in each iteration
    eta=0.1;
    figure;
    iteration=1;
    while 1
        iteration
        disp('Learning rate=====>');
        disp(eta);
        
        % check for each input data for misclassification, if misclassified
        % try to minimize w
        for i=1:1:80
            
            % update the weights for misclassified response
            if((X(i,:).*Y(i))*w'<=0)
                w=w+eta*(X(i,:).*Y(i));
                plot(class_0(:,1),class_0(:,2),'rO',class_1(:,1),class_1(:,2),'bX');

                % plot the decision boundary for each data point
                db_line=drawline(w(2:end),w(1));
                db_line.Color='black';
                db_line.LineWidth=2;

                title('PLOT 2: PERCEPTRON LEARNING ALGORITHM');
                xlabel('X1 ---->');
                ylabel('X2 ---->');
                legend('y=-1','y=1');

            end
            
            pause(0.001);
        end
        
        % stop after learning rate is sufficient enough to settle the
        % decision boundary
        if(eta<0.001)
            break;
        end
        eta=eta*0.8;
        iteration=iteration+1;
    end
    toc
end