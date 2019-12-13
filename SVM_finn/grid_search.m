function [q,C] = grid_search(features_xy,labels,k)

% n x n grid
n = 3;

q_grid = linspace(1,21,n);
C_grid = linspace(1,61,n);

tic

evals = zeros(n,n,3);

for i = 1:n
    for j = 1:n
        fprintf('## i=%i, j=%i ##\n', i, j);
        svm_results = inner_kfold_trainer(C_grid(i), q_grid(j),k,features_xy,labels);
        evals(i,j,:) = mean(svm_results(:,:));
        % precision only
        %evals(i,j,:) = max(svm_results(:,1));
    
        toc
    end
end

f = evals;

% retrieving the best value of the hyper parameters, to use in the outer
% fold
[M1,I1] = max(f);
[~,I2] = max(M1(1,1,:));
index = I1(:,:,I2);
C = C_grid(index(1))
q = q_grid(index(2))

%plot(q_grid, evals(:,:,1));
%hold on;
%plot(q_grid, evals(:,:,2));
%plot(q_grid, evals(:,:,3));