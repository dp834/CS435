%{
    Damien Prieur
    CS 435
    Assignment 1
%}

%% Global Setup


img_original = double(imread('images/Lenna.png'));
output_location_prefix = 'images/generated/';
%cleanup any previously generated images
delete('images/generated/*');
fprintf('Cleaned "images/generated"\n');

%% Question 2
% convert image to grayscale using this formula
% Gray=0.2989R+0.5870*G+0.1140B

grayscale = 0.2989 * img_original(:,:,1) + ...
            0.5870 * img_original(:,:,2) + ...
            0.1140 * img_original(:,:,3);

imwrite(uint8(grayscale), strcat(output_location_prefix, 'Q2_grayscale.png'))

fprintf('Done with question 2\n');

%% Question 3
% Produce three binary images for each of the following thresholds
% 25%, 50%, 75%

thresholds = [.25,.50,.75];

for threshold = thresholds
    %get the sum of all the values
    threshold_image = img_original(:,:,1) + ...
                      img_original(:,:,2) + ...
                      img_original(:,:,3);
    threshold_image = threshold_image > (threshold * 255 * 3);
    threshold_image = threshold_image * 255;
    filename = sprintf('%sQ3_threshold%d%%.png', ...
                       output_location_prefix, uint8(threshold*100));
    imwrite(uint8(threshold_image), filename);

end

fprintf('Done with question 3\n');

%% Question 4
% Make a hisogram for the grayscale image as well as one for each color channel
% To plot the histogram use the "bar" function


bins = zeros(4,256);
flat = [ uint8(reshape(grayscale          , 1 ,numel(grayscale)))          ; ...
         uint8(reshape(img_original(:,:,1), 1 ,numel(img_original(:,:,1)))); ...
         uint8(reshape(img_original(:,:,2), 1 ,numel(img_original(:,:,2)))); ...
         uint8(reshape(img_original(:,:,3), 1 ,numel(img_original(:,:,3)))) ];

for bin = 0:255
    bins(:,bin+1) = sum(flat==bin, 2);
end
bins = bins/numel(grayscale);

chart = bar(bins(1,:));
set(chart(1), 'FaceColor', 'black', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(gcf, strcat(output_location_prefix, 'Q4_hist_gray.png'));

chart = bar(bins(2,:));
set(chart(1), 'FaceColor', 'red', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(gcf, strcat(output_location_prefix, 'Q4_hist_red.png'));

chart = bar(bins(3,:));
set(chart(1), 'FaceColor', 'green', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(chart, strcat(output_location_prefix, 'Q4_hist_green.png'));

chart = bar(bins(4,:));
set(chart(1), 'FaceColor', 'blue', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(gcf, strcat(output_location_prefix, 'Q4_hist_blue.png'));

chart = bar(0:255, bins(2:end,:));
set(chart(1), 'FaceColor', 'red', 'EdgeColor', 'none', 'BarWidth', 1);
set(chart(2), 'FaceColor', 'green', 'EdgeColor', 'none', 'BarWidth', 1);
set(chart(3), 'FaceColor', 'blue', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(gcf, strcat(output_location_prefix, 'Q4_hist_rgb.png'));

chart = bar(0:255, bins);
set(chart(1), 'FaceColor', 'black', 'EdgeColor', 'none', 'BarWidth', 1);
set(chart(2), 'FaceColor', 'r', 'EdgeColor', 'none', 'BarWidth', 1);
set(chart(3), 'FaceColor', 'g', 'EdgeColor', 'none', 'BarWidth', 1);
set(chart(4), 'FaceColor', 'b', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(gcf, strcat(output_location_prefix, 'Q4_hist_all.png'))

fprintf('Done with question 4\n');

%% Question 5
% Using the grayscale image preform contrast stretching
% use the histogram to inform decision

%function is at the bottom of the file
img_contrast_stretched = arrayfun(@contrast_stretch_func, grayscale);

bins = zeros(1,256);
flat = uint8(reshape(img_contrast_stretched, 1 ,numel(img_contrast_stretched)));

for bin = 0:255
    bins(bin+1) = sum(flat==bin);
end
bins = bins/numel(img_contrast_stretched);

chart = bar(0:255, bins);
set(chart(1), 'FaceColor', 'black', 'EdgeColor', 'none', 'BarWidth', 1);
saveas(gcf, strcat(output_location_prefix, 'Q5_hist_contrast_stretched.png'));
imwrite(uint8(img_contrast_stretched), strcat(output_location_prefix, 'Q5_grayscale_contrast_stretched.png'))
fprintf('Done with question 5\n');


% perform simple contrast stretch by removing the any empty regions
% for our image we are doing [26,245] -> [0,255]
function y = contrast_stretch_func(x)
    y = uint8((255-0)/(230-30)*(x-30) + 0);
    if y < 0
        y = 0;
    elseif y > 255
        y = 255;
    end
end
