  
q_grid = linspace(1,19,10);
C_grid = linspace(1,59,10);
tic
evals = zeros(10,10,3);
for i = 1:10
    for j = 1:10
        fprintf('## i=%i, j=%i ##\n', i, j);
        svm_results = kfold_polynomial_trainer(C_grid(i), q_grid(j));
        evals(i,j,:) = mean(svm_results(:,:));
    
        toc
    end
end

f = evals;

%plot(q_grid, evals(:,:,1));
%hold on;
%plot(q_grid, evals(:,:,2));
%plot(q_grid, evals(:,:,3));