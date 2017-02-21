function [ ] = analysis()

load('data4students.mat');
im = datasetInputs{1};
im1 = im(55,:);
im2 = reshape(im1, 30, 30);
disp(im1);
colormap gray;
imagesc(im2);


end

