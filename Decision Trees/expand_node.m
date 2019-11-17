function [parent] = expand_node(features,labels,parent)
%this function takes a node and the features & labels which belong to it,
%assign its left child and right child to it when it returns
%a node is struct('index', 'null', 'Thereshold','null','left', 'null', 'right', 'null','hasNext',true);
[index,Best_Threshold,Best_Gain] = id3(features,labels);
if index == 0
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

    for i = 1 : m
        if features(i,index)< Best_Threshold
            nextleftdata=[nextleftdata;features(i,:)];
            nextleftlabel=[nextleftlabel;labels(i,:)];
        end
        if features(i,index)>Best_Threshold
            nextrightdata=[nextrightdata;features(i,:)];
            nextrightlabel=[nextrightlabel;labels(i,:)];
        end
    end
    %create left child
    leftnode = struct('index','null', 'Thereshold','null','left', 'null', 'right', 'null', 'features','null','labels','null','hasNext',true);
    leftnode.features=nextleftdata;
    leftnode.labels=nextleftlabel;
    %create right child
    rightnode = struct('index','null', 'Thereshold','null','left', 'null', 'right', 'null', 'features','null','labels','null','hasNext',true);
    rightnode.features=nextrightdata;
    rightnode.labels=nextrightlabel;
    %assign the children back to the parent node
    parent.left = leftnode;
    parent.right = rightnode;
    
end