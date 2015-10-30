%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% October 23, 2014
% CS229
% PS2 plot1.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function calculates learned coefficients
% Input:
%   machine.ascii
%       X     - Nxd dimensional input matrix
%       Y_hat - Nx1 response vector
%   lambda- the complexity parameter that controls shrinkage
% Output:
%   beta_series - learned coefficients for given values of lambda
% Plot:
%   Learned coefficients vs shrinkage parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = plot1()
    
    % tic-toc to find the exection time of code
    tic
    
    test_X=[]; train_X=[];
    load -ascii machine.ascii
    machine=[ones(size(machine,1),1) machine];
    [m n]=size(machine);
    lambda=[10^-3:100:10^3];
    [lambda_m lambda_n] = size(lambda);
    flambda=[];
    for l=lambda
        ridge_arr=[];
        for ex=1:20
            % calculate the training and testing set
            % randomly generate row indices for training set that will be 20%
            % of input data
            train_rows=randperm(m,ceil(0.2*m));
            test_rows=[];
    
            % Add remaining 80% data to testing dataset
            i=1;
            while(i<=size(machine,1))
        
                % if the index not in training set, include in testing set
                if(~ismember(i,train_rows))
                    test_rows=[test_rows i];
                end
                i=i+1;
            end
    
            train_data=machine(train_rows,:);
            test_data=machine(test_rows,:);
            
            % Normalize the input train data (except for Y)
            train_X=bsxfun(@rdivide,bsxfun(@minus,train_data(:,1:7),mean(train_data(:,1:7))),std(train_data(:,1:7)));
            train_X(:,1)=1;
            
            % Apply training normalization to test data (except for Y)
            test_X=bsxfun(@rdivide,bsxfun(@minus,test_data(:,1:7),mean(train_data(:,1:7))),std(train_data(:,1:7)));
            test_X(:,1)=1;
 
            % find a best fit model from training data
            %beta_series=[];
        
            % "Elements of Statistical Learning", equation (3.44)
            Id=[eye(size(train_X,2))];
            Id(1,1)=0;
            beta_series=inv((train_X'*train_X)+l*Id)*(train_X'*train_data(:,8));
                             
            % calculate the average squared error between predicted and
            % expected responses. Penalty not added here as the problem
            % asks only for squared error
            ridge=(sum((test_data(:,8)-(test_X*beta_series)).^2))/size(test_X,1);
            ridge_arr=[ridge_arr ridge];
        end
        flambda=[flambda mean(ridge_arr)];
    end
    figure;
    semilogx(lambda,flambda);
    title('PLOT 1: Ridge Performance with varying \lambda');
    xlabel('df(\lambda)');
    ylabel('Ridge Performance');
    toc
end