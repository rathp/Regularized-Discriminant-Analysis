clc; clear;
%RDA Implementation
Data_set_RDA=importdata('data_cancer.mat');
X_train1=Data_set_RDA.X;
Y_train1=Data_set_RDA.Y;

mapMatrix1=horzcat(X_train1,Y_train1);
[rows1,columns1]=size(mapMatrix1);


%part 1(D)

%fix random seed
s=RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
savedState=s.State;

%separating the data into 150 training and 66 testing
f_d=randperm(rows1,150);
s.State=savedState;
map_temp_d=mapMatrix1(f_d(1:150),(1:columns1));
X_train=map_temp_d(:,(1:(columns1-1)));
Y_train=map_temp_d(:,columns1);

%setting the number of classes
numofClass=length(unique(Y_train));

b_d=1:rows1;
c_d=ismember(b_d,f_d);
j_d=1;


d_d=zeros(1,66);
for i=1:rows1
if c_d(i)==0
   d_d(j_d)=i;
   j_d=j_d+1;
end
end
map_test1_d=mapMatrix1(d_d(1,:),(1:columns1));

%separating the test data
X_test=map_test1_d(:,(1:(columns1-1)));
Y_test=map_test1_d(:,columns1);

j=1;
pred_rda=zeros(66,19);

%putting the predictions in a cell and finding predictions 
x=0.1:0.05:1;
for i=0.1:0.05:1
       [RDAmodel(1,j)]=rathp_RDA_train(X_train,Y_train,i,numofClass);
       pred_rda(:,j)=rathp_RDA_test(X_test, RDAmodel(1,j), numofClass);
%        pred_rda1(:,j)=rathp_RDA_test(X_train, RDAmodel(1,j), numofClass);
       j=j+1;
end

[rows_rdapred,cols_rdapred]=size(pred_rda);

Y_test=single(Y_test);
confmat_cell=cell(cols_rdapred,1);
% confmat_cell1=cell(cols_rdapred,1);
pred_rda=single(pred_rda);
% pred_rda1=single(pred_rda1);

%computing the CCR using cells
for i=1:cols_rdapred
    confmat_cell{i,1}=confusionmat(pred_rda(:,i),Y_test);
    ccr(i)=trace(confmat_cell{i,1})./(sum(sum(confmat_cell{i,1})));
%     confmat_cell1{i,1}=confusionmat(pred_rda1(:,i),Y_train);
%     ccr1(i)=trace(confmat_cell1{i,1})./(sum(sum(confmat_cell1{i,1})));
end

figure;
plot(x,ccr);
xlabel('lambda');
ylabel('CCR');
title('CCR of RDA over various Lambda values');
