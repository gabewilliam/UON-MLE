answer1=features_xy_flds(1);
train_data=answer1{1}(:,:);

answer2=labels_flds(1);
train_label=answer2{1}(:,1);
%id3(train_data,train_label);

%new_node=tree(train_data,train_label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%outer loop
root=struct('index', 1, 'Thereshold','null','left', 'null', 'right', 'null','hasNext',true,'features','null','labels','null');
root.features = train_data;
root.labels = train_label;
root=expand(root);

%id3(nextright.features,nextright.labels);



function [node] = expand(node)
if node.hasNext == false
    return;
end
cn = unique(node.labels);
i = length(cn);

if i == 1
    return;
end

node=expand_node(node.features, node.labels, node);
left=expand_node(node.left.features,node.left.labels,node.left);
[n,m] = size(node.left.features);
if n == 0
    return;
end
right=expand_node(node.right.features,node.right.labels,node.right);
[x,y] = size(node.right.features);
if x == 0
    return;
end
node.left = expand(left);
node.right = expand(right);
end

function [parent] = expand_node(features,labels,parent)
%this function takes a node and the features & labels which belong to it,
%assign its left child and right child to it when it returns
%a node is struct('index', 'null', 'Thereshold','null','left', 'null', 'right', 'null','hasNext',true);
if ~parent.hasNext
    return;
end

[index,Best_Threshold,Best_Gain] = id3(features,labels);
if index <= 1
    parent.hasNext=false;
    return;
end
%fprintf('Atribute %i at threshold %.2f gives the greatest information gain of %.3f ', [index, Best_Threshold, Best_Gain]);
[m,n1]=size(features); %n1=98
[~,n2]=size(labels);  %n2=5

nextleftdata=[];
nextleftlabel=[];
nextrightdata=[];
nextrightlabel=[];

emptyleft=true;
emptyright=true;


    for i = 1 : m
        if features(i,index)< Best_Threshold
            emptyleft=false;
            nextleftdata=[nextleftdata;features(i,:)];
            nextleftlabel=[nextleftlabel;labels(i,:)];
        end
        if features(i,index)>=Best_Threshold
            emptyright=false;
            nextrightdata=[nextrightdata;features(i,:)];
            nextrightlabel=[nextrightlabel;labels(i,:)];
        end
    end
    
    %create left child
    leftnode = struct('index','null', 'Thereshold','null','left', 'null', 'right', 'null', 'features','null','labels','null','hasNext',true);
    leftnode.features=nextleftdata;
    leftnode.labels=nextleftlabel;
    if length(unique(leftnode.labels))==1
        leftnode.hasNext=false;
    end
    %create right child
    rightnode = struct('index','null', 'Thereshold','null','left', 'null', 'right', 'null', 'features','null','labels','null','hasNext',true);
    rightnode.features=nextrightdata;
    rightnode.labels=nextrightlabel;
    if length(unique(rightnode.labels))==1
       rightnode.hasNext=false;
    end
    
    if m == 0
        leftnode.hasNext=false;
    end
    
    if m == 0
        rightnode.hasNext=false;
    end
    
    
    %assign the children back to the parent node
    parent.left = leftnode;
    parent.right = rightnode;
    
end

% The code takes atributes and labels as inputs and outputs
% the best atribute with best threshold value.
% For each atribute: 
% The labels will be sorted in ascending order of that atribute's features
% then we split the labels at each index and calculate the corresponding
% gain. We then find that the best gain occurs at a split at the ith element.
% the corresponding threshold value will then be between the ith and the i+1 
% element in the features array


%looping through attributes

function [index, Best_Threshold, Best_Gain] = id3(atributes, labels)
Best_Gain = 0;
Best_Threshold = 0;
index = 0;

[m,n] = size(atributes);

if m<=1
    return;
end

for i = 1:n
    [Gain, threshold] = th(atributes(:,i),labels);
    if Gain > Best_Gain
        Best_Gain = Gain;
        Best_Threshold = threshold;
        index = i;
    end
end

fprintf('Atribute %i at threshold %.2f gives the greatest information gain of %.3f\n', [index, Best_Threshold, Best_Gain]);

end


function [best_gain, threshold, index,features_sorted] = th(features,labels)
%finding best threshold for given atribute

[labels_sorted, features_sorted] = sorting(labels, features);
[p,n] = pn(labels_sorted);

best_gain = 0;
%index = 2;
index =1;

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

%threshold = (features_sorted(index) + features_sorted(index + 1))/2;
if isempty(features_sorted)
    return;
else 
    threshold = features_sorted(index);
end
 
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
% ordering the arrays with respect to the atribute at stake
linked = cat(2,features,labels);
sorted = sortrows(linked,1);
%labels_sorted = sorted(:,2);
[~,n]=size(linked);
labels_sorted = sorted(:,n);
features_sorted = sorted(:,1);

end
