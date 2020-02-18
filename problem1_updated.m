clear all
close all
clc
%%
load ('data_batch_2.mat');
%%
load ('batches.meta.mat');


%% Building the model
% I am just using 100 samples in total
% You have to used at least 100 samples from each class
sample_num = 1000;

% Set these based on number of features you want
% For this example I am only using 14 features
nH = 8; nS = 4; nV = 2;

Histogram = zeros(sample_num, nH+nS+nV);
for i = 1:sample_num
    % convert to hsv format
    image = data(i,:);
    image_rgb = reshape(image, [32,32,3]);
    %figure(1); subplot(sqrt(sample_num),sqrt(sample_num),i); imshow(image_rgb);
    image_hsv = rgb2hsv(image_rgb);
    image_hsv = reshape(image_hsv, [32*32,3]);
    
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

% I am averaging the feature of each class as our model class
model_label = labels(1:100,:);
for i=0:9
    ind = find(model_label==i);
    model_histogram(i+1,:) = mean(Histogram(ind,:));  % This the equivalent to table that is mentioned in the question
end

% Plotting one histogram example
figure(2);
subplot(1,3,1);bar(v1,h1);xlabel('H');
subplot(1,3,2);bar(v2,h2);xlabel('S');
subplot(1,3,3);bar(v3,h3);xlabel('V');

%% test image
load ('test_batch.mat');
j = 7;
image = data(j,:);
image_rgb = reshape(image, [32,32,3]);
figure(3);
imshow(image_rgb);

% Create a function instead.

image_hsv = rgb2hsv(image_rgb);
image_hsv = reshape(image_hsv, [32*32,3]);

[h1, v1] = hist(image_hsv(:,1),c1);
[h2, v2] = hist(image_hsv(:,2),c2);
[h3, v3] = hist(image_hsv(:,3),c3);   
test_histogram = [h1, h2, h3];

for i=1:10
    dist(i) = pdist2(test_histogram, model_histogram(i,:));
end
[Y,I] = min(dist);

fprintf("Predicted: %d\n", I-1)
fprintf("Actual: %d\n",labels(j))