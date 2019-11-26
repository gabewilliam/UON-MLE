
Mdl = fitcsvm(Name,Value, 'KernelFunction','linear', 'BoxConstraint',1,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',true));
