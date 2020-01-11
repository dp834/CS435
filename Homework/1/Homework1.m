%{
    Damien Prieur
    CS 435
    Assignment 1
%}

%% Global Setup


img_original = double(imread('../../Images/Lenna.png'));
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
    threshold_image = threshold_image > double(threshold * 255 * 3);
    threshold_image = threshold_image * 255;
    filename = sprintf('%sQ3_threshold%d%%.png', ...
                       output_location_prefix, uint8(threshold*100));
    imwrite(uint8(threshold_image), filename);

end

fprintf('Done with question 3\n');
