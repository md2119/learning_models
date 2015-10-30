%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: Mandar Darwatkar
% SID: 861141010
% October 15, 2014
% CS229
% PS1 part-a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tic-toc to find the exection time of code
tic
% order=N= number points in one example to be sampled
order=[50 100 500];
fd=[];
for N=order
    
    %dim=D= dimensionality of spaces
    dim=[1:10:100];
    d_avg_dist=[];
    for D=dim
        
        % M is the number of samples to be taken for vectors of D dimensions
        M=150;    
        sample_avg=[];
        while(M)
            
            % following snippet calculates uniformly distributed random
            % points from unit hyperball.
            pts=randn(N,D);
            pts=pts.*repmat((rand(N,1).^(1/D))./sqrt(sum(pts.*pts,2)),[1 D]);
            
            % following snippet finds the average minimum euclieadean
            % distance of each vector with every other vector in given
            % sample.
            pts1=repmat(permute(pts',[3 1 2]),[N 1 1]);
            pts=repmat(pts,1,1,N);
            pts=(pts-pts1).^2;
            clearvars pts1
            dist=sqrt(sum(pts,2)); %*(pts-pts1),2));
            clearvars pts
            dist(isnan(0./dist))=inf(1);
            avg_dist=mean(min(dist));
            
            % average distance for M samples is stored for later average to
            % be taken.
            sample_avg=[sample_avg avg_dist];
            M=M-1;
        end
        
        % Store f(D)
        d_avg_dist=[d_avg_dist mean(sample_avg)];
    end
    
    %Store f(D) for different values of N
    fd=[fd; d_avg_dist];
    disp('N='); disp(N);
    
end

% Plot f(D) i.e. proxmity of points taken from unit hyperball against
% number of dimensions
figure;
plot(dim,fd(1,:),'r');
hold on
plot(dim,fd(2,:),'b');
plot(dim,fd(3,:),'g');
hold off
title({'Curse of Dimensionality','Proximity of point vs dimensionality of space'});
xlabel('DIMENSIONALITY OF SPACE(D)');
ylabel('AVG. DIST. BETWEEN NEAREST NEIGHBOURS F(D)');
legend('N=50','N=100','N=500');
toc
%EOF