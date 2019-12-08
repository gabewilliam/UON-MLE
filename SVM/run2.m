sigma_grid = linspace(130, 180, 50);
tic
for i = 1:50

    fprintf('## i=%i ##\n', i);
    svm_results = kfold_grbf_trainer(sigma_grid(i));
    evals(i,:) = mean(svm_results(:,:));
    
    toc
    
end

f = evals;

plot(sigma_grid, evals(1:50,1));
hold on;
plot(sigma_grid, evals(1:50,2));
plot(sigma_grid, evals(1:50,3));