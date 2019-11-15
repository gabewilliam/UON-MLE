% The code takes atributes and labels as inputs and outputs
% the best atribute with best threshold value.
% For each atribute: 
% The labels will be sorted in ascending order of that atribute's features
% than we split the labels at each index and calculate the corresponding
% gain. We than find that the best gain occurs at a split at the ith element.
% the corresponding threshold value wil than be between the ith and the i+1 
% element in the features array


%looping through attributes

function [index, Best_Threshold, Best_Gain] = id3(atributes, labels)
Best_Gain = 0;
Best_Threshold = 0;
index = 0;

[~,n] = size(atributes);

for i = 1:n
    [Gain, threshold] = th(atributes(:,i),labels);
    if Gain > Best_Gain
        Best_Gain = Gain;
        Best_Threshold = threshold;
        index = i;
    end
end

fprintf('Atribute %i at threshold %.2f gives the greatest information gain of %.3f ', [index, Best_Threshold, Best_Gain]);

end


function [best_gain, threshold, index,features_sorted] = th(features,labels)
%finding best threshold for given atribute

[labels_sorted, features_sorted] = sorting(labels, features);
[p,n] = pn(labels_sorted);

best_gain = 0;
index = 2;

for i=1:length(labels_sorted)
    
%splitting data in to lower values than threshold and including the
%threshold value (at index i)
   data_lower = labels_sorted(1:i);
   
   [p_i,n_i] = pn(data_lower);
   gain = G(p,n,p_i,n_i);
   
   if gain > best_gain
       best_gain = gain;
       index = i;
   end
 
end

% threshold value is somewhere inbetween the two feature values
threshold = (features_sorted(index) + features_sorted(index + 1))/2;
   
end


function [p,n] = pn(input)
%input = each column of attribute, e.g. labels_au1
n=0;
p=0;
for i=1:length(input)
    if input(i)==0
        n=n+1;
    else
        p=p+1;
    end
end
end

function entropy = I(p,n)
entropy = -(p/(p+n))*log2(p/(p+n))-(n/(p+n))*log2(n/(p+n));
end

function remainder_i = A_i(p,n,p_i,n_i)
remainder_i = I(p_i,n_i)*(p_i+n_i)/(p+n);
end

function gain = G(p,n,p_i,n_i)
p_j = p - p_i;
n_j = n - n_i;
gain = I(p,n) - A_i(p,n,p_i,n_i)- A_i(p,n,p_j,n_j); 
end

function [labels_sorted, features_sorted] = sorting(labels, features)
% odrering the arrays with respect to the atribute at stake
linked = cat(2,features,labels);
sorted = sortrows(linked,1);
labels_sorted = sorted(:,2);
features_sorted = sorted(:,1);
end