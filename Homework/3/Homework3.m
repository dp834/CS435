%{
    Damien Prieur
    CS 435
    Assignment 3
%}

%% Global Setup


output_location_prefix = 'images/generated/';
%cleanup any previously generated images
delete('images/generated/*');
fprintf('Cleaned "images/generated"\n');

image_1_location = 'images/Lenna.png';
original_img_1 = imread(image_1_location);

image_2_location = 'images/aloeR.jpg';
original_img_2 = imread(image_2_location);


%% Question 2
% Take two images and resize them to a size specified by some parameters
% Resize each image using:
% 1) nearest neighbor sampling
% 2) linear interpolation

new_width = 1000;
new_height = 1000;

img = nearest_neighbor_resize(original_img_1, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_1_nearest_1.png'));

img = linear_interpolation_resize(original_img_1, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_1_linear_1.png'));

img = nearest_neighbor_resize(original_img_2, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_2_nearest_1.png'));

img = linear_interpolation_resize(original_img_2, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_2_linear_1.png'));

new_width = 600;
new_height = 600;

img = nearest_neighbor_resize(original_img_1, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_1_nearest_2.png'));

img = linear_interpolation_resize(original_img_1, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_1_linear_2.png'));

img = nearest_neighbor_resize(original_img_2, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_2_nearest_2.png'));

img = linear_interpolation_resize(original_img_2, new_width, new_height);
imwrite(uint8(img), strcat(output_location_prefix, 'Q2_img_2_linear_2.png'));


%% Question 3
% Get grayscale image
% Smooth image using a gaussian filter
% Get the gradients
% Pad the image so that the size doesn't change
% Compute the energy of each pixel and show the resulting image

energy_img_1 = get_energy_img(original_img_1);
imwrite(uint8(energy_img_1), strcat(output_location_prefix, 'Q3_img_1_energy.png'));


energy_img_2 = get_energy_img(original_img_2);
imwrite(uint8(energy_img_2), strcat(output_location_prefix, 'Q3_img_2_energy.png'));



%% Question 4
% Optimal seams

test_energy_mat = [
2 3	4 5 1;
1 0 2 2 1;
4 3 5 1 2;
4 4 4 4 6;
4 5 2 0 2;
2 3 3 0 3;
];

seam = find_optimal_seam(energy_img_1);
highlighted_seam = highlight_pixels(original_img_1, seam, [255 0 0]);
imwrite(uint8(highlighted_seam), strcat(output_location_prefix, 'Q4_img_1_highlighted_seam.png'));

seam = find_optimal_seam(energy_img_2);
highlighted_seam = highlight_pixels(original_img_2, seam, [255 0 0]);
imwrite(uint8(highlighted_seam), strcat(output_location_prefix, 'Q4_img_2_highlighted_seam.png'));

%% Question 5
% Seam Carving
% Create a video with the current optimal seam highlighted
% Then remove that seam and repeat until the image is empty
% Pad the image so that it remains the same size

tmp = seam_carving_video(original_img_1, strcat(output_location_prefix, 'Q5_seam_carving_video.avi'));


%% Functions

% nearest neighbor resize
function img = nearest_neighbor_resize(input_img, new_width, new_height)

    orig_height = size(input_img,1);
    orig_width = size(input_img,2);

    % create empty image of new size
    img = zeros(new_height, new_width, 3);

    for x = 1:new_width
        for y = 1:new_height
            src_x = round(x*orig_width/new_width);
            src_y = round(y*orig_height/new_height);

            if(src_x < 1)
                src_x = 1;
            end

            if(src_x > orig_width)
                src_x = orig_width;
            end

            if(src_y < 1)
                src_y = 1;
            end

            if(src_y > orig_height)
                src_y = orig_height;
            end

            img(y,x,:) = input_img(src_y, src_x, :);
        end
    end
end

% linear interpolation resize
function img = linear_interpolation_resize(input_img, new_width, new_height)
    orig_height = size(input_img,1);
    orig_width = size(input_img,2);

    % create empty image of new size
    img = zeros(new_height, new_width, 3);
    for x = 1:new_width
        for y = 1:new_height

            % Where we would sample from if data existed
            ideal_src_x = x*orig_width/new_width;
            ideal_src_y = y*orig_height/new_height;

            % special cases if we are blowing the image up, will effectively pad the border
            if(ideal_src_x < 1)
                ideal_src_x = 1;
            end

            if(ideal_src_x > orig_width)
                ideal_src_x = orig_width;
            end

            if(ideal_src_y < 1)
                ideal_src_y = 1;
            end

            if(ideal_src_y > orig_height)
                ideal_src_y = orig_height;
            end

            % perfect match use it
            if(ideal_src_x == floor(ideal_src_x) && ideal_src_y == floor(ideal_src_y))
                img(y,x,:) = input_img(ideal_src_y, ideal_src_x,:);
                continue
            end

            % if we have a perfect match on just a row or column it will double count
            % the values but will also divide by the right amount, so the only special
            % case is when we have a perfect match on our source point

            % 4 points closest to where we want
            sample_points = [floor(ideal_src_x) floor(ideal_src_y) ;
                             floor(ideal_src_x) ceil(ideal_src_y)  ;
                             ceil(ideal_src_x)  floor(ideal_src_y) ;
                             ceil(ideal_src_x)  ceil(ideal_src_y)] ;

            % find distance of each point from target point
            distances = abs(sample_points - [ideal_src_x ideal_src_y]);
            distances = distances .* distances;
            distances = distances(:,1) + distances(:,2);
            distances = arrayfun(@(x) sqrt(x), distances);

            new_val = double(zeros(1,1,3));

            % compute weighted sum for linear interpolation of all channels at once
            for i = 1:size(sample_points, 1)
                tmp = distances(i) * input_img(sample_points(i,2), sample_points(i,1),:);
                new_val = new_val + double(tmp);
            end
            new_val = 1/sum(distances) * new_val;

            img(y,x,:) = uint8(new_val);
        end
    end
end

function energy_img = get_energy_img(original_img)
    % using a 5x5 gaussian filter with sigma = 1
    gaussian_filter_convolution = [
            0.003765    0.015019    0.023792    0.015019    0.003765;
            0.015019    0.059912    0.094907    0.059912    0.015019;
            0.023792    0.094907    0.150342    0.094907    0.023792;
            0.015019    0.059912    0.094907    0.059912    0.015019;
            0.003765    0.015019    0.023792    0.015019    0.003765;
        ];


    partial_x_convolution = [
    1/3 0 -1/3 ;
    1/3 0 -1/3 ;
    1/3 0 -1/3 ; ];

    partial_y_convolution = [
     1/3  1/3  1/3 ;
      0    0    0  ;
    -1/3 -1/3 -1/3 ; ];

    orig_grayscale = double(rgb2gray(original_img));
    smoothed_img = conv2(orig_grayscale, gaussian_filter_convolution, 'same');

    partial_x_img = conv2(smoothed_img, partial_x_convolution, 'same');
    partial_y_img = conv2(smoothed_img, partial_y_convolution, 'same');
    energy_img = abs(partial_x_img) + abs(partial_y_img);
end

% Finds optimal seam given energy image as input, returns list of columns representing seam
function seam = find_optimal_seam(energy_img)
    seam = zeros(2,size(energy_img,1));
    seam(1,:) = 1:size(seam,2);

    % Solve via the dynamic programming approach by finding the min cost to reach a pixel from the pixels above it
    cumulative_energy = zeros(size(energy_img));
    % -1 means came from parent above and to the left, 0 means parent above, 1 means parent above and to the right
    path_matrix = zeros(size(energy_img));

    % First row is just copied
    cumulative_energy(1,:) = energy_img(1,:);

    for y = 2:size(energy_img, 1)
        % check borders first so that ther are no special cases and can just loop through the rest

        %checking left border
        if(energy_img(y-1,1) < energy_img(y-1,2))
            cumulative_energy(y,1) = cumulative_energy(y-1,1) + energy_img(y,1);
            path_matrix(y,1) = 0;
        else
            cumulative_energy(y,1) = cumulative_energy(y-1,2) + energy_img(y,1);
            path_matrix(y,1) = 1;

        end

        if(energy_img(y-1,end-1) < energy_img(y-1,end))
            cumulative_energy(y,end) = cumulative_energy(y-1,end-1) + energy_img(y,end);
            path_matrix(y,end) = -1;
        else
            cumulative_energy(y,end) = cumulative_energy(y-1,end) + energy_img(y,end);
            path_matrix(y,end) = 0;

        end

        for x = 2:(size(energy_img, 2) - 1)
            if(energy_img(y-1, x-1) < energy_img(y-1,x) && energy_img(y-1,x-1) < energy_img(y-1, x+1))
                cumulative_energy(y, x) = cumulative_energy(y-1, x-1) + energy_img(y,x);
                path_matrix(y,x) = -1;
            elseif(energy_img(y-1, x) < energy_img(y-1, x+1))
                cumulative_energy(y, x) = cumulative_energy(y-1, x) + energy_img(y,x);
                path_matrix(y,x) = 0;
            else
                cumulative_energy(y,x) = cumulative_energy(y-1, x+1) + energy_img(y,x);
                path_matrix(y,x) = 1;
            end
        end
    end

    % find min at the bottom row and use the path_matrix to trace path upwards
    min_weight_path = min(cumulative_energy(end,:));
    min_weight_ending_index = find(cumulative_energy(end,:)==min_weight_path,1);
    seam(2,end) = min_weight_ending_index;

    for i = (size(seam,2) -1):-1:1
       seam(2,i) = seam(2,i+1) + path_matrix(i,seam(2,i+1));
    end
end

% Given a matrix of points make all thos pixels red
function img = highlight_pixels(original_img, pixels, color)
    img = original_img;
    for i = pixels
       img(i(1),i(2),:) = color;
    end
end

function video = seam_carving_video(original_img, filename)
    img = original_img;
    %video = VideoWriter(filename);
    %open(video);

    for i = 1:size(original_img,2)
        energy = get_energy_img(img);
        seam = find_optimal_seam(energy);
        img = highlight_pixels(img, seam, [255 0 0]);
        frame = pad_columns(img, size(original_img,2));
        %writeVideo(video, frame);
        img = remove_pixels(img, seam);

    end

    %close(video);

end

function cropped_img = remove_pixels(original_img, pixels)
    cropped_img = reshape(original_img,1,numel(original_img(:,:,1)),3);
    for pixel = pixels
        cropped_img(pixel(1) * pixel(2) - pixel(1) + 1, :) = [];
    end
    cropped_img = reshape(cropped_img, size(original_img,1), size(original_img, 2) -1, 3);
end

function padded_img = pad_columns(original_img, target_width)
    padded_img = zeros(size(original_img,1), target_width, 3);
    columns_to_add = target_width - size(original_img, 2);
    left_pad  = ceil(columns_to_add / 2);
    right_pad = target_width - floor(columns_to_add / 2);

    padded_img(left_pad+1:right_pad, :, :) = original_img;
end

