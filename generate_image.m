clc;
clear all;
close all;

data = {'City', 'Forest', 'Snow mountain'};
save_dir = './LR';
mkdir(save_dir);

for i = 1 : length(data)
    file_list = dir([data{i}, '/*.jpeg']);
    % file_list = file_list.name;
    for j = 1 : length(file_list)
        im = imread(['./', data{i}, '/', file_list(j).name]);
        im_lr = imresize(im, 1/4, 'bicubic');
        imwrite(im_lr, [save_dir, '/', data{i}, '_lr', num2str(j), '.jpeg']);
    end
end

