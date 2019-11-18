answer1=features_xy_flds(1);
train_data=answer1{1}(1,:);
answer2=labels_flds(1);

number=8290; %the number of features to be tested, change this

train_label=answer2{1}(1:number,1);
output = classify(train_data, dTree);
array=zeros(number,1);
difference=0;

for i = 1:number
    array(i)=classify(answer1{1}(i,:),dTree);
end

array=cat(2,array,train_label);

for i = 1:number
    if array(i,1)~=array(i,2)
        difference = difference+1;
    end
end

correct_rate = 1-(difference/number);











