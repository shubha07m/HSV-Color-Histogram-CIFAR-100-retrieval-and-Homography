clear all
close all
clc
%% loading one of the training batches
load ('data_batch_1.mat');
%% loading class labels for the training data
load ('batches.meta.mat');


%% Building the model
% We are using 100 from the batch 
sample_num = 100;

% This example I am only using 14 features
nH = 8; nS = 4; nV = 2;
%%
Histogram = zeros(sample_num, nH+nS+nV);
for i = 1:sample_num
    % convert to hsv format
    image = data(i,:);
    image_rgb = reshape(image, [32,32,3]);
    image_hsv = rgb2hsv(image_rgb);
    image_hsv = reshape(image_hsv, [32*32,3]);
    %%
    % normalization
    c1 = [1:nH]*(1/(nH+1));
    c2 = [1:nS]*(1/(nS+1));
    c3 = [1:nV]*(1/(nV+1));
    
    % histogram on each channel
    [h1, v1] = hist(image_hsv(:,1),c1);
    [h2, v2] = hist(image_hsv(:,2),c2);
    [h3, v3] = hist(image_hsv(:,3),c3);   
    Histogram(i,:) = [h1, h2, h3];
end
%%
% Taking the average of each 14 features one by one and storing it in model_histogram
model_label = labels(1:100,:); % labaling all 100 images
for i=0:9
    ind = find(model_label==i);
    model_histogram(i+1,:) = mean(Histogram(ind,:));  % This the equivalent to table that is mentioned in the question
end
%%
% Plotting single histogram for only one image
figure(1);
subplot(1,4,1);bar(v1,h1);xlabel('H');
subplot(1,4,2);bar(v2,h2);xlabel('S');
subplot(1,4,3);bar(v3,h3);xlabel('V');

%% loading a single image from test batch to predict the class
load ('test_batch.mat');
j = 7;
image = data(j,:);
image_rgb = reshape(image, [32,32,3]);
figure(2);
imshow(image_rgb);
%%
image_hsv = rgb2hsv(image_rgb);
image_hsv = reshape(image_hsv, [32*32,3]);

[h1, v1] = hist(image_hsv(:,1),c1);
[h2, v2] = hist(image_hsv(:,2),c2);
[h3, v3] = hist(image_hsv(:,3),c3);   
test_histogram = [h1, h2, h3];
%%
for i=1:10
    dist(i) = pdist2(test_histogram, model_histogram(i,:));
end
%%
[Y,I] = min(dist);

fprintf("Predicted: %d\n", I-1)
fprintf("Actual: %d\n",labels(j))