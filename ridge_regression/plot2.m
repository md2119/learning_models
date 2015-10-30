%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name: Mandar Darwatkar
% SID: 861141010
% October 23, 2014
% CS229
% PS2 plot2.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function plots the value of each learned coefficient (and the bias
% coefficient) as a function of lambda, each as a separate curve on the
% same plot.
% Input:
%   machine - 209x7 matrix where 209x6 represents input and 7th column 
%   (209x1) represents expected response
% Output:
%   Learning coefficients
% Plot:
%   Coefficients vs shrinkage parameter
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot2()
    tic
    % get the input and expected response as separate entities
    load -ascii machine.ascii
    machine=[ones(size(machine,1),1) machine];
    [m n]=size(machine);
    lambda=[10^-3:100:10^3];
    X=machine(:,1:7);
    Y=machine(:,8);

    
    % normalze the input features, z-score normalization
    X=bsxfun(@rdivide,bsxfun(@minus,X,mean(X)),std(X));
    X(:,1)=1;
    
    % calulate other learned coefficients
    beta_series=[];
    for i=lambda
        
        % "Elements of Statistical Learning", equation (3.44)
        I=[eye(size(X,2))];
        I(1,1)=0;
        beta=inv((X'*X)+i*I)*(X'*Y);
        beta_series=[beta_series beta];
    end
    
    % Plot bias and other coefficients as function of lambda
    figure;
    semilogx(lambda,beta_series,'LineWidth',2);
    title('PLOT 2: Profile of ridge coefficients with varying \lambda');
    xlabel('df(\lambda)');
    ylabel('Coefficients');
    %legend('\beta0','\beta1','\beta2','\beta3','\beta4','\beta5','\beta6');
    legend('bias','machine cycle time', 'min main memory size','max main memory size', 'cache memory size','min no. of channels','max no. of channels');
    toc
end
