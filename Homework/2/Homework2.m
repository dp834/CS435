%{
    Damien Prieur
    CS 435
    Assignment 1
%}

%% Global Setup


output_location_prefix = 'images/generated/';
%cleanup any previously generated images
delete('images/generated/*');
fprintf('Cleaned "images/generated"\n');

%% Theory Questions
% Question 1 apply a 3x3 mean filter to the matrix

theory_1_original_mat = [
7 7 6 3 3 4 2 2 ; ...
3 7 2 6 4 4 5 7 ; ...
5 4 7 5 1 1 2 2 ; ...
2 1 3 4 1 3 5 6 ; ...
6 2 2 7 4 2 5 4 ; ...
2 2 2 3 6 6 6 7 ; ...
4 6 5 6 7 3 4 1 ; ...
5 2 4 6 1 4 1 4 ; ];

mean_kernel = 1/9 * ones(3,3);

convoluted_matrix = conv2(theory_1_original_mat, mean_kernel, 'valid')

% Question 2 apply partial derivatives to the matrix

theory_2_original_mat = [
7 7 6 ; ...
3 7 2 ; ...
5 4 7 ; ];

partial_x_convolution = [
1/3 0 -1/3 ; ...
1/3 0 -1/3 ; ...
1/3 0 -1/3 ; ];

partial_y_convolution = [
 1/3  1/3  1/3    ; ...
  0    0    0     ; ...
-1/3 -1/3 -1/3    ; ];

partial_x = conv2(theory_2_original_mat, partial_x_convolution, 'valid')
partial_y = conv2(theory_2_original_mat, partial_y_convolution, 'valid')

%% Question 2 Plotting pixel value vs log exposure
% Read file images/memorial/images.txt
% Load in each picture with its associated exposure time
images_file = fopen('images/memorial/images.txt', 'r');

% string is filename and float is exposure time
format_spec = '%s %f';
comment = '#';

images_to_read = textscan(images_file, format_spec, 'CommentStyle', comment);
% have a 1x2 cell (matrix) with {1}{1} being names and {1}{2} being exposure times
filenames = cell2mat(images_to_read{1}(:));
exposures = images_to_read{2};
%images    = zeros(numel(filenames(:,1)));
images = 'null';
size(images)
for img = 1:numel(filenames(:,1))
    img_read_in = uint8(imread(strcat('images/memorial/',filenames(img,:))));
    if ischar(images) == true
        image_size = size(img_read_in);
        images = zeros(numel(filenames(:,1)), image_size(1), image_size(2), image_size(3));
    end
    images(img,:,:,:) = img_read_in;
end

% Select three pixel locations and plot the values in the red channel as a function of exposure time

% Pixels i will be looking at (100,100), (200,200), (300,300)
pixel_locations  = [ 100 100; 200 200; 300 300];

data = zeros(3,numel(exposures));
data(1,:) = images(:,pixel_locations(1,1), pixel_locations(1,2), 1);
data(2,:) = images(:,pixel_locations(2,1), pixel_locations(2,2), 1);
data(3,:) = images(:,pixel_locations(3,1), pixel_locations(3,2), 1);


plot(exposures, data(1,:),'-o');
hold on;
plot(exposures, data(2,:),'-o');
plot(exposures, data(3,:),'-o');
legend('100,100', '200,200','300,300');
hold off;
saveas(gcf, strcat(output_location_prefix, 'Q2_exposure_vs_intensity.png'));

%% Question 3
% Find the log irradiance function g(z_{ij)} for each color channel. then repeat plot from previous section


% possible pixel values = 256 (0-255)

vals = 4;
pixel_loc = [1 1; 2 2];
images_a = zeros(3,2,2);
images_a(1, 1,1) = 0;
images_a(2, 1,1) = 1;
images_a(3, 1,1) = 2;
images_a(1, 2,2) = 3;
images_a(2, 2,2) = 3;
images_a(3, 2,2) = 3;
exposures_tmp = [.5 1 2];


asdf = log_irradiance_inverse(vals, pixel_loc, exposures_tmp, images_a)



function mapping = log_irradiance_inverse(color_values, pixel_locations, exposures, images)
    A = zeros(size(pixel_locations,1)*size(exposures,2) + 1, color_values + size(pixel_locations,1));    
    b = zeros(size(A,2),1);

    k = 1;

    for loc = 1:size(pixel_locations,1)
        for exposure = 1:size(exposures,2)
            z = images(exposure,pixel_locations(loc,1),pixel_locations(loc,2));
            A(k,z+1) = 1;
            A(k,loc + color_values) = -1;
            b(k) = log(exposures(exposure));

            k = k +1;
        end
    end

    A(k, round(color_values/2)) = 1;
    b(k) = 0;

    x = A\b;
    mapping = x(1:color_values);
end


