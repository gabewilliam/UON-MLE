sigma_grid = linspace(50, 150, 50);
C_grid = linspace(1,100,50);
tic
evals = zeros(50,50,3);
for i = 1:50
    for j = 1:50
        fprintf('## i=%i, j=%i ##\n', i, j);
        svm_results = kfold_grbf_trainer(C_grid(i), sigma_grid(j));
        evals(i,j,:) = mean(svm_results(:,:));
    
        toc
    end
end

f = evals;

plot(sigma_grid, evals(:,1));
hold on;
plot(sigma_grid, evals(:,2));
plot(sigma_grid, evals(:,3));