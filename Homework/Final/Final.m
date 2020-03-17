%{
    Damien Prieur
    Partner: Daniel Schwartz
    CS 435
    Final
%}

%% Global Setup

format short
global OUTPUT_LOCATION_PREFIX OCTAVE_COUNT SCALE_COUNT REDUCTION_FACTOR;
OUTPUT_LOCATION_PREFIX = 'images/generated/';
OCTAVE_COUNT = 4;
SCALE_COUNT = 5;
REDUCTION_FACTOR = 2;  % value is squared so images are subsampled to 1/4

% Cleanup any previously generated images
delete('images/generated/*');
fprintf('Cleaned "images/generated"\n');

% Read images
image_1_location = 'images/im1.jpg';
im1 = imread(image_1_location);
image_2_location = 'images/im2.jpg';
im2 = imread(image_2_location);




%% Question 1 - Hard Coding Point Correspondences

% Define hardcoded point correspondences (source -> target)
source = [1710, 1237; ...
          1672, 3103; ...
          1456, 1145; ...
          2805, 3304];
target = [ 546, 1228; ...
           553, 3173; ...
           264, 1117; ...
          1652, 3208];

% Draw hardcoded point correspondences
mark_hardcoded_points(im1, im2, source, target)




%% Question 2 - Compute Transformation Matrix, Project, and Blend!

% Stitch images together using hardcoded point correspondences (source -> target)
hardcoded_stitched_image = stitch_images(im1, im2, source, target, @linear_interpolation);
imwrite(hardcoded_stitched_image, strcat(OUTPUT_LOCATION_PREFIX, 'final_manual_stitched.png'));
%stitched_image = stitch_images(right_img, left_img, target, source, @linear_interpolation);
%stitched_image = stitch_images(left_img, right_img, source, target, @fast_interp);
%stitched_image = stitch_images(right_img, left_img, target, source, @fast_interp);




%% Question 3 - Create Scale-Space Image Pyramids

fprintf('Starting Scale Space Pyramid\n');
img1_scale_space = compute_scale_space(im1);
visualize_scale_space(img1_scale_space)
img2_scale_space = compute_scale_space(im2);




%% Question 4 - Finding the Local Maximas

fprintf('Starting DoG/maxima img1\n');
img1_DoG = compute_dog(img1_scale_space);
img1_local_maxima = compute_local_maxima(im1, img1_DoG);
draw_extrema_points(img1_local_maxima, im1, 'final_all_extrema_left.png');

fprintf('Starting DoG/maxima img2\n');
img2_DoG = compute_dog(img2_scale_space);
img2_local_maxima = compute_local_maxima(im2, img2_DoG);
draw_extrema_points(img2_local_maxima, im2, 'final_all_extrema_right.png');

fprintf('Starting pruning, rest is fast\n');
img1_key_points = prune_unstable_maxima(im1, img1_local_maxima, 60);
img2_key_points = prune_unstable_maxima(im2, img2_local_maxima, 60);
draw_extrema_points(img1_key_points, im1, 'final_pruned_extrema_left.png');
draw_extrema_points(img2_key_points, im2, 'final_pruned_extrema_right.png');




%% Question 5 - Keypoint Description and Matching

% Find matching keypoints: For each keypoint compute a feature vector and look for a match in the other set of keypoints
img1_feature_vectors = get_features_from_key_points(im1, img1_key_points);
img2_feature_vectors = get_features_from_key_points(im2, img2_key_points);

% Initialize lists
key_point_correspondences = [];
img1_pairs = zeros(size(img1_feature_vectors, 2),2);
img2_pairs = zeros(size(img2_feature_vectors, 2),2);

% Match keypoints
for i = 1:size(img1_pairs,1)
    img1_pairs(i,:) = [i sort_by_similarity(img1_feature_vectors(:,i), img2_feature_vectors)];
end
for i = 1:size(img2_pairs,1)
    img2_pairs(i,:) = [sort_by_similarity(img2_feature_vectors(:,i), img1_feature_vectors) i];
    if(img1_pairs(img2_pairs(i,1),2) == i)
        key_point_correspondences(end+1, :) = [img2_pairs(i,1), i];
    end
end

% Draw both images and mark the key point correspondences
img1_key_point_correspondences = [];
img2_key_point_correspondences = [];
for i = 1:size(key_point_correspondences,1)
    if abs(img1_key_points(key_point_correspondences(i,1),2) - img2_key_points(key_point_correspondences(i,2),2)) < .10*size(im1, 1)
        if (size(im1,2) - img1_key_points(key_point_correspondences(i,1),1) + img2_key_points(key_point_correspondences(i,2),1)) < (size(im1,2) + size(im2, 2))/2
            img1_key_point_correspondences(end+1, :) = img1_key_points(key_point_correspondences(i,1),:);
            img2_key_point_correspondences(end+1, :) = img2_key_points(key_point_correspondences(i,2),:);
        end
    end
end

% Draw lines of keypoint correspondences
draw_point_correspondences(im1, im2, img1_key_point_correspondences, img2_key_point_correspondences, 'final_automatic_point_correspondences.png');


%% Question 6 - Find the Transformation Matrix via RANSAC and Stitch

% Run RANSAC and get transformation points
[img1_points, img2_points] = ransac(img1_key_point_correspondences, img2_key_point_correspondences, log(1-.9)/log(1-.05^4));
draw_point_correspondences(im1, im2, img1_points, img2_points, 'final_selected_point_correspondences.png');

% Stitch images and write to file
automatic_stitched_image = stitch_images(im1, im2, img1_points, img2_points, @linear_interpolation);
imwrite(automatic_stitched_image, strcat(OUTPUT_LOCATION_PREFIX, 'final_automatic_stitched.png'));

%% Extra credit stitch multiple images, this is slow sorry
imgs = {'images/7.jpg', 'images/6.jpg', 'images/5.jpg', 'images/4.jpg', 'images/3.jpg', 'images/2.jpg', 'images/1.jpg', 'images/0.jpg'};
for i = 1:size(imgs,2)
    imgs{i} = imread(imgs{i});
end
img_left  = [];
img_right = imgs{1};

for i = 2:size(imgs,2)
    img_left  = img_right;
    img_right = imgs{i};

    [left_key_points, right_key_points] = get_key_point_correspondeces(img_left,img_right);
    H = compute_transformation_matrix(left_key_points, right_key_points);
    [new_image_size, origin] = get_combined_size(img_left, img_right, H);

    % Save to img_right so that the loop will work
    img_right = merge_images_multipass(img_left, img_right, H, new_image_size, origin, @fast_interp);
end
%

img_right(img_right == -1) = 0;
cool_img = img_right;
imshow(cool_img);

% Functions



% Problem 1 Functions

% Marks point correspondences hardcoded
function mark_hardcoded_points(im1, im2, source_pts, target_pts)
    global OUTPUT_LOCATION_PREFIX

    % Show images
    height_pad = size(im1, 1) - size(im2, 1);
    if(height_pad < 0)
        im1(end:end-height_pad, :,:) = zeros([-height_pad, size(im1,2), size(im1,3)]);
    elseif(height_pad > 0)
        im2(end:end+height_pad, :,:) = zeros([height_pad, size(im2,2), size(im2,3)]);
    end
    imshow([im1 im2]);

    % Define point colors
    colors = ['r', 'y', 'g', 'b'];

    % Draw circles at all point correspondences
    for i=1:size(source_pts,1)
        draw_circle(source_pts(i,1),source_pts(i,2),25,colors(i),1);
    end
    for i=1:size(target_pts,1)
        draw_circle(target_pts(i,1)+size(im1,2),target_pts(i,2),25,colors(i),1);
    end

    % Save image
    saveas(gcf, strcat(OUTPUT_LOCATION_PREFIX, 'final_hardcoded_points.png'));
    close(gcf);
end


% Plot circle
% Input: Location (x,y) radius, color, and boolean that determines if circle should be filled
% Output: Plotted circle at desired location with desired attributes
function draw_circle(x,y,radius,color,fill_it)
    hold on
    th = linspace(0,2*pi,100);
    if fill_it
        fill(x + radius*cos(th), y + radius*sin(th), color, 'LineStyle', 'none', 'EdgeColor', 'none');
    else
        plot(x + radius*cos(th), y + radius*sin(th), color);
    end
    hold off
    axis equal
end




% Problem 3 Functions

% Computes Scale-Space Image Pyramid
% Input: Original Image, Number of Octaves, Number of Scales, Subsampling Factor
% Output: Cell Array of Images in Scale-Space Image Pyramid
function scale_space = compute_scale_space(im)
    global OCTAVE_COUNT SCALE_COUNT REDUCTION_FACTOR
    img = rgb2gray(im);
    scale_space = cell(OCTAVE_COUNT,SCALE_COUNT);
    sigma_o = 1.6;
    for octave=1:OCTAVE_COUNT
        for scale=1:SCALE_COUNT
            sigma = sigma_o*2^(octave-1)*sqrt(2)^(scale-1);
            gauss_kernel = gaussian_filter(ceil(3*sigma), sigma);
            smoothed_gray = conv2(img, gauss_kernel, 'same');
            scale_space{octave, scale} = uint8(smoothed_gray);
            sigma = sigma * sqrt(2);
        end
        img = img(1:REDUCTION_FACTOR:end, 1:REDUCTION_FACTOR:end, :);
    end
end


% Write scale sapce to image file
function visualize_scale_space(scale_space)
    global OCTAVE_COUNT SCALE_COUNT OUTPUT_LOCATION_PREFIX
    figure;
    for y=1:OCTAVE_COUNT
        for x=1:SCALE_COUNT
            subplot(OCTAVE_COUNT, SCALE_COUNT, SCALE_COUNT*(y-1) + x);
            imshow(scale_space{y, x});
            axis on;
        end
    end
    saveas(gcf, strcat(OUTPUT_LOCATION_PREFIX, 'final_scale_space.png'));
    close(gcf);
end




% Problem 4 Functions

% Computes Difference of Gaussian in Image Pyramid
% Input: Cell Array of Images in Scale-Space Image Pyramid
% Output: Difference of Gaussians across different Octaves in Cell Array
function DoG = compute_dog(scale_space)
    global OCTAVE_COUNT SCALE_COUNT
    DoG = cell(OCTAVE_COUNT,SCALE_COUNT-1);
    for octave=1:OCTAVE_COUNT
        for scale=1:SCALE_COUNT-1
            DoG{octave, scale} = scale_space{octave, scale+1} - scale_space{octave, scale};
        end
    end
end


% Computes Local Maxima of a DoG
% Input: Cell Array containg DoG across different octaves
% Output: Binary matrix of local maxima
function output_img = compute_local_maxima(img, DoG)
    global OCTAVE_COUNT SCALE_COUNT
    output_img = zeros(size(img,1),size(img,2));

    for octave=1:OCTAVE_COUNT
        for scale=1:SCALE_COUNT-1
            DoG{octave, scale} = padarray(DoG{octave, scale}, [1, 1]);
        end
    end

    for octave=1:OCTAVE_COUNT
        scale_factor = 2^(octave-1);
        DoG_octave = zeros(size(DoG{octave, 1}, 1), size(DoG{octave, 1},2), 2 + SCALE_COUNT);

        for scale=1:SCALE_COUNT-1
            DoG_octave(:,:,scale+1) = DoG{octave, scale};
        end

        for x = 2:size(DoG_octave,2)-1
            for y = 2:size(DoG_octave,1)-1
                for z = 2:size(DoG_octave,3)-1
                    % Look at the cube around the point and if it's the max then we mark it as a max
                    if(3*3+5 == find(DoG_octave(y-1:y+1,x-1:x+1,z-1:z+1) == max(DoG_octave(y-1:y+1,x-1:x+1,z-1:z+1), [], 'all')))
                        output_img((y-1)*(scale_factor-1)+1, (x-1)*(scale_factor-1)+1) = 1;
                    end
                end
            end
        end

    end
end


% Converts binary image to vector of coordinates
function coords = convert_matrix_to_coords(mat)
    [ys, xs] = find(mat);
    coords = [xs, ys];
end


% Draw circles at extrema
function draw_extrema_points(mat, img, fname)
    global OUTPUT_LOCATION_PREFIX
    [x,y] = find(mat==1);
    imshow(uint8(img));
    radius=0.5;
    hold on;
    for k=1:size(x,1)
        theta=0:0.01:2*pi;
        xval=radius*cos(theta);
        yval=radius*sin(theta);
        plot(y(k)+xval,x(k)+yval, 'r');
    end
    hold off;
    saveas(gcf, strcat(OUTPUT_LOCATION_PREFIX, fname));
    close(gcf);
end


function key_points = prune_unstable_maxima(img, candidates,threshold)
    % Compute Edge Pixels
    edge_points = edge(rgb2gray(img));

    % The patch will go this count in all directions to make the patch
    % So a WIDTH of 4 will be a 9x9 patch
    WIDTH = 4;
    % Since i will be adding each color channel divide by 3
    CONTRAST_THRESH = threshold*3;

    % Remove all points that are too close to the edge to get a contrast reading
    candidates(1:WIDTH,:) = 0;
    candidates(:,1:WIDTH) = 0;
    candidates(end-WIDTH:end,:) = 0;
    candidates(:,end-WIDTH:end) = 0;

    % Canditate point is on an edge we throw it out
    candidates(candidates == edge_points) = 0;
    [ys, xs] = find(candidates);

    for i = 1:numel(ys)
        % find the std_dev of brightness for each color channel
        std_dev = std(double(img(ys(i)-WIDTH:ys(i)+WIDTH, xs(i)-WIDTH:xs(i)-WIDTH, :)), 0, [1 2]);
        if(sum(std_dev, 'all') < CONTRAST_THRESH)
           candidates(ys(i), xs(i)) = 0;
        end
    end

    [row,col] = find(candidates);
    key_points = [col, row];
end




% Problem 5 Functions

% Extracts features from image given keypoints
function features = get_features_from_key_points(img, key_points)
    xs = key_points(:,1);
    ys = key_points(:,2);
    %img = padarray(img, [9 9]);
    features = zeros(9*9*3, numel(xs));
    for i = 1:numel(xs)
        features(:, i) = reshape(img(ys(i)-4:ys(i)+4, xs(i)-4:xs(i)+4, :), [9*9*3, 1]);
    end
end


% Given an array of features, a feature to compare to, and similarity_func will return a ordered list based on similarity
function most_similar = sort_by_similarity(source_fv, tests_fv)
    % index into original list, similarity
    similarities = zeros(size(tests_fv,2), 1);
    for item = 1:size(tests_fv,2)
        similarities(item) = histogram_similarity(tests_fv(:,item), source_fv);
    end
    [~, most_similar] = min(similarities);

end


% Compute similarity of two histogram feature vectors
function similarity = histogram_similarity(hist_a, hist_b)
    similarity = 0;
    for bin = 1:size(hist_a,1)
        similarity = similarity + abs(hist_a(bin) - hist_b(bin));
    end
end


% Draw point correspondences and write to file
function draw_point_correspondences(im1, im2, source_pts, target_pts, fname)
    global OUTPUT_LOCATION_PREFIX

    height_pad = size(im1, 1) - size(im2, 1);
    if(height_pad < 0)
        height_pad
        im1(end+1:end+abs(height_pad), :,:) = zeros([abs(height_pad), size(im1,2), size(im1,3)]);
    elseif(height_pad > 0)
        im2(end+1:end+height_pad, :,:) = zeros([height_pad, size(im2,2), size(im2,3)]);
    end

    imshow([im1 im2]);
    for i=1:size(source_pts,1)
        draw_line(source_pts(i,:), target_pts(i,:), size(im1,2));
    end
    saveas(gcf, strcat(OUTPUT_LOCATION_PREFIX, fname));
    close(gcf);
end


% Draw line between two coordinates (coord1,coord2) and displaces by width
function draw_line(coord1, coord2, width)
    hold on
    line([coord1(1) (coord2(1)+width)], [coord1(2) coord2(2)],'Color', 'red');
    hold off
    axis equal
end




% Problem 6 Functions

% Find optimal keypoint correspondence using RANdom SAmpling Consensus
function [kp_source_p, kp_target_p]= ransac(kp_source, kp_target, iter_count)
    min_dist = exp(1000); % Infinity
    NUM_POINTS = 4;
    random_source = zeros(NUM_POINTS,2);
    random_target = zeros(NUM_POINTS,2);
    for iter=1:iter_count
        % Select `num_points` from source and targeted keypoints
        random_inds = randperm(size(kp_source, 1), NUM_POINTS);
        for i=1:NUM_POINTS
            random_source(i,:) = kp_source(random_inds(i),:);
            random_target(i,:) = kp_target(random_inds(i),:);
        end
        % Compute transformation matrix from random points
        h_mat = compute_transformation_matrix(random_source, random_target);
        % Project source points using transformation matrix
        kp_projected = map_multiple_points(kp_source, h_mat);
        % Compute distance from projected to target points
        euclid_dist = sum(abs(kp_projected - kp_target), 'all');
        % Check if distance is smaller than previous min distance
        if euclid_dist < min_dist
            min_inds = random_inds;
            min_dist = euclid_dist;
        end
    end
    kp_source_p = kp_source(min_inds,:);
    kp_target_p = kp_target(min_inds,:);
end


% Maps a vector of points, source_vector given a homography transformation matrix, H
function projected_points = map_multiple_points(source_vector, H)
    % Append column of ones to act as `z`
    z_col = ones(size(source_vector, 1), 1);
    source_vector = [source_vector z_col];

    % Project source points using transformation matrix
    source_vector = H * (source_vector');
    xs = source_vector(1, :);
    ys = source_vector(2, :);
    ws = source_vector(3, :);

    % Scale projected points correctly
    xs = xs ./ ws;
    ys = ys ./ ws;
    projected_points(:,1) = xs;
    projected_points(:,2) = ys;
end


% Maps a single point, p given a homography transformation matrix, H
function output = map_point(H, p)
    output = H*p;
    output = [output(1)/output(3), output(2)/output(3)];
end




% Homography Functions

function H = compute_transformation_matrix(source_points, target_points)
    if(size(source_points) ~= size(target_points))
        fprintf('Source and target points do not have the same number of elements');
        H = [];
        return
    end

    x     = source_points(:,1);
    y     = source_points(:,2);
    x_hat = target_points(:,1);
    y_hat = target_points(:,2);

    A = zeros(2*size(source_points,1),9);
    for i = 1:size(A,1)/2
        A(2*i-1:2*i, :) = [ -x(i), -y(i), -1,     0,     0,  0, x_hat(i)*x(i), x_hat(i)*y(i), x_hat(i); ...
                                0,     0,  0, -x(i), -y(i), -1, y_hat(i)*x(i), y_hat(i)*y(i), y_hat(i)];
    end

    % d/dh (h'A'Ah) => 2A'Ah = 0
    [~, ~, V] = svd(A'*A);
    V(:,end);
    % use values of the smallest eigenvalue
    H = reshape(V(:,end),[3, 3])';
end




% Stitching Functions

% Computes combined size of two images given homography transformtaion matrix
function [new_size, origin] = get_combined_size(img1, img2, H)
    % Find where the corners map to
    % Corners ul ur ll lr
    corners = [            1,             1, 1; ...
               size(img1, 2),             1, 1; ...
                           1, size(img1, 1), 1; ...
               size(img1, 2), size(img1, 1), 1;];
    % transpose of corners cause I guess i write the coordinates weirdly
    corners = H * (corners');
    xs = corners(1, :);
    ys = corners(2, :);
    ws = corners(3, :);

    % get the correctly scaled x and y values
    xs = xs ./ ws;
    ys = ys ./ ws;

    % we find the origin here because I don't want to recompute the corner points

    % one of the 8 corners is going to be mapped to a corner
    % if we know which corner is mapped to its corner then we can work backwards to find the
    % origin of the output plane
    %     1-ul   1-ur   1-ll   1-lr   2-ul   2-ur        2-ll              2-lr
    xs = [xs(1), xs(2), xs(3), xs(4),  1, size(img2, 2),            1, size(img2, 2)];
    ys = [ys(1), ys(2), ys(3), xs(4),  1,             1, size(img2,1), size(img2, 1)];

    [minx, ~] = min(xs);
    minxis = find(xs==minx);
    [maxx, ~] = max(xs);
    maxxis = find(xs==maxx);
    [miny, ~] = min(ys);
    minyis = find(ys==miny);
    [maxy, ~] = max(ys);
    maxyis = find(ys==maxy);

    new_size = [uint32(maxy-miny), uint32(maxx-minx), 3];
    if(isempty(intersect(minxis, minyis)) == 0)
        origin = [-minx, -miny];
    elseif(isempty(intersect(minxis, maxyis)) == 0)
        origin = [-minx, double(new_size(1)) - maxy];
    elseif(isempty(intersect(maxxis, minyis)) == 0)
        origin = [double(new_size(2)) - maxx, -miny];
    elseif(isempty(intersect(maxxis, maxyis)) == 0)
        origin = [double(new_size(2)) - maxx, double(new_size(1)) - maxy];
    elseif(isempty(intersect(minxis, minyis)) == 0)
        origin = [-minx, -miny];
    else
        fprintf('Error finding origin, Non linear transform!??!?!?\n');
        return
    end

%{
    % Can be shorter 100% TODO rewrite
    if(xs(1) < xs(2) && xs(1) < xs(3) && xs(1) < xs(4) && ys(1) < ys(2) && ys(1) < ys(3) && ys(1) < ys(4))    %if the upper left corner is staying in the corner then the the origin is at the coordinate
        origin = [-xs(1), -ys(1)];
    elseif(xs(2) < xs(3) && xs(2) < xs(4) && ys(2) > ys(1) && ys(2) > ys(3) && ys(2) > ys(4))
        origin = [-xs(1), new_size(2) - ys(1)];
    elseif(xs(2) < xs(3) && xs(2) < xs(4) && ys(3) < ys(1) && ys(3) < ys(2) && ys(3) < ys(4))
        origin = [new_size(1) - xs(3), ys(3)];
    else
        origin = [new_size(1) - xs(4), new_size(2) - ys(4)];
    end
%}
    origin = ceil(origin);
end


% Computes merged image from two images, homograpy matrix, origin, and interpolation function
function merged_img = merge_images(img1, img2, H, new_size, origin, interpolation_function)
    % create output matrix
    merged_img = uint8(zeros(new_size));
    origin = double(origin);

    % get our inverse transformation
    H_inv = inv(H);

    % used to mark if pixel is empty not just black, will be used if stitching many images together
    EMPTY_PIXEL = [-1,-1,-1];
    for y = 1:size(merged_img,1)
        % Gives an idea of where we are at
        if(mod(y, 500) == 0)
            fprintf("y = %d : %.02f%%\n", y, y/double(new_size(1)));
        end
        for x = 1:size(merged_img,2)

            %iterating over our range
            range_point = [x, y] - origin;
            domain_point = map_point(H_inv, [range_point(1); range_point(2); 1]);

            in_range  = range_point(1) >= 1 && range_point(1) <= size(img2, 2) && range_point(2) >= 1 && range_point(2) <= size(img2, 1);
            %if(in_range && isEqual(img2(range_point(2), range_point(1), :), EMPTY_PIXEL))
            %    in_range = false;
            %end

            in_domain = domain_point(1) >= 1 && domain_point(1) <= size(img1, 2) && domain_point(2) >= 1 && domain_point(2) <= size(img1, 1);
            % TODO domain points are not a specific point, need to think about handling this differently...
            %if(in_domain && isEqual(img1(domain_point(2), domain_point(1), :), EMPTY_PIXEL))
            %    in_domain = false;
            %end

            if(in_domain && in_range)
                % if location is in both then blend with method of choice

                % TODO make blending a function pass, may need to place all points first then blend with im2
                % need to talk with Dan

                % going to do simple alpha blending with alpha = .5

                % if point is between locations then interpolate
                alpha = .5;
                new_val = interpolation_function(img1, domain_point)*alpha;
                % range values should not need interpolation as x,y and origin are integers
                new_val = new_val + img2(range_point(2), range_point(1), :)*(1-alpha);
            elseif(in_domain)
                % if location is in our left image and not in right just take left
                % if point is between locations then interpolate
                new_val = interpolation_function(img1, domain_point);
            elseif(in_range)
                % if location is in our right image and not in left just take right
                % range values should not need interpolation as x,y and origin are integers
                new_val = img2(range_point(2), range_point(1), :);

            else
                % if location is in neither then set to black (0 0 0)
                % mark as not from either, will be mapped to zero
                new_val = EMPTY_PIXEL;
            end
            merged_img(y,x,:) = new_val;
        end
    end
    % TODO if we do multiple we need to not do this until the last one
    % map points that were unfilled to [0,0,0];
    merged_img(merged_img == -1) = 0;
end


% Stitches two images from source and target images and interpolation function
function stitched_img = stitch_images(img1, img2, keypoint_src, keypoint_target, interpolation_func)
    H = compute_transformation_matrix(keypoint_src, keypoint_target);
    [new_image_size, origin] = get_combined_size(img1, img2, H);
    stitched_img = merge_images(img1, img2, H, new_image_size, origin, interpolation_func);
end




% Linear Interpolation Functions

function value = linear_interpolation(img, loc)
     sample_points = [floor(loc(1)) floor(loc(2)) ;
                      floor(loc(1)) ceil(loc(2) ) ;
                      ceil(loc(1) ) floor(loc(2)) ;
                      ceil(loc(1) ) ceil(loc(2) )];

     % find distance of each point from target point
     distances = abs(sample_points - [loc(1) loc(2)]);
     distances = distances .* distances;
     distances = distances(:,1) + distances(:,2);
     distances = arrayfun(@(x) sqrt(x), distances);

     value = double(zeros(1,1,3));
     % compute weighted sum for linear interpolation of all channels at once
     for i = 1:size(sample_points, 1)
         tmp = (distances(i)/sum(distances)) * img(sample_points(i,2), sample_points(i,1),:);
         value = value + double(tmp);
     end
     value = uint8(value);
end


function value = fast_interp(img, loc)
    loc = floor(loc);
    loc(loc == 0) = 1;
    value = img(loc(2), loc(1), :);
end


% Computes Gaussian Filter
function f=gaussian_filter(n,s)
    x = -1/2:1/(n-1):1/2;
    [Y,X] = meshgrid(x,x);
    f = exp( -(X.^2+Y.^2)/(2*s^2) );
    f = f / sum(f(:));
end

% Stitches two images from source and target images and interpolation function
% input is a cell array containing all the images
function stitched_img = stitch_many_images(imgs)
    if(size(imgs,2) < 2)
        stitched_img = imgs{1};
        return
    end
    img_left  = [];
    img_right = imgs{1};

    for i = 2:size(imgs,2)
        img_left  = img_right;
        img_right = imgs{i};

        [left_key_points, right_key_points] = get_key_point_correspondeces(img_left,img_right);
        H = compute_transformation_matrix(left_key_points, right_key_points);
        [new_image_size, origin] = get_combined_size(img_left, img_right, H);

        % Save to img_right so that the loop will work
        img_right = merge_images_multipass(img_left, img_right, H, new_image_size, origin, @fast_interp);
    end
    img_right(img_right == -1) = 0;
    stitched_img = img_right;

end

function [img1_points, img2_points] = get_key_point_correspondeces(img1, img2)
    img1_scale_space = compute_scale_space(img1);
    img2_scale_space = compute_scale_space(img2);


    img1_DoG = compute_dog(img1_scale_space);
    img1_local_maxima = compute_local_maxima(img1, img1_DoG);

    img2_DoG = compute_dog(img2_scale_space);
    img2_local_maxima = compute_local_maxima(img2, img2_DoG);

    img1_key_points = prune_unstable_maxima(img1, img1_local_maxima, 0);
    img2_key_points = prune_unstable_maxima(img2, img2_local_maxima, 0);

    % Find matching keypoints: For each keypoint compute a feature vector and look for a match in the other set of keypoints
    img1_feature_vectors = get_features_from_key_points(img1, img1_key_points);
    img2_feature_vectors = get_features_from_key_points(img2, img2_key_points);

    % Initialize lists
    key_point_correspondences = [];
    img1_pairs = zeros(size(img1_feature_vectors, 2),2);
    img2_pairs = zeros(size(img2_feature_vectors, 2),2);

    % Match keypoints
    for i = 1:size(img1_pairs,1)
        img1_pairs(i,:) = [i sort_by_similarity(img1_feature_vectors(:,i), img2_feature_vectors)];
    end
    for i = 1:size(img2_pairs,1)
        img2_pairs(i,:) = [sort_by_similarity(img2_feature_vectors(:,i), img1_feature_vectors) i];
        if(img1_pairs(img2_pairs(i,1),2) == i)
            key_point_correspondences(end+1, :) = [img2_pairs(i,1), i];
        end
    end

    % Draw both images and mark the key point correspondences
    img1_key_point_correspondences = [];
    img2_key_point_correspondences = [];
    for i = 1:size(key_point_correspondences,1)
        if abs(img1_key_points(key_point_correspondences(i,1),2) - img2_key_points(key_point_correspondences(i,2),2)) < .10*size(img1, 1)
            if (size(img1,2) - img1_key_points(key_point_correspondences(i,1),1) + img2_key_points(key_point_correspondences(i,2),1)) < (size(img1,2) + size(img2, 2))/2
                img1_key_point_correspondences(end+1, :) = img1_key_points(key_point_correspondences(i,1),:);
                img2_key_point_correspondences(end+1, :) = img2_key_points(key_point_correspondences(i,2),:);
            end
        end
    end
    draw_point_correspondences(img1, img2, img1_key_point_correspondences, img2_key_point_correspondences, 'asdfsadf.png');

    [img1_points, img2_points] = ransac(img1_key_point_correspondences, img2_key_point_correspondences, log(1-.9)/log(1-.05^4));
end

function merged_img = merge_images_multipass(img1, img2, H, new_size, origin, interpolation_function)
    % create output matrix
    merged_img = uint8(zeros(new_size));
    origin = double(origin);

    % get our inverse transformation
    H_inv = inv(H);

    % used to mark if pixel is empty not just black, will be used if stitching many images together
    EMPTY_PIXEL = [-1,-1,-1];
    for y = 1:size(merged_img,1)
        % Gives an idea of where we are at
        if(mod(y, 500) == 0)
            fprintf("y = %d : %.02f%%\n", y, y/double(new_size(1)));
        end
        for x = 1:size(merged_img,2)

            %iterating over our range
            range_point = [x, y] - origin;
            domain_point = map_point(H_inv, [range_point(1); range_point(2); 1]);

            in_range  = range_point(1) >= 1 && range_point(1) <= size(img2, 2) && range_point(2) >= 1 && range_point(2) <= size(img2, 1);

            in_domain = domain_point(1) >= 1 && domain_point(1) <= size(img1, 2) && domain_point(2) >= 1 && domain_point(2) <= size(img1, 1);
            
            if(in_domain && in_range)
                % if location is in both then blend with method of choice

                % TODO make blending a function pass, may need to place all points first then blend with im2
                % need to talk with Dan

                % going to do simple alpha blending with alpha = .5

                % if point is between locations then interpolate
                alpha = .5;
                domain_val = interpolation_function(img1, domain_point);
                if(domain_val < 0)
                   alpha = 0; 
                end
                new_val = domain_val*alpha;
                if(new_val < 0)
                    if(domain_val < 0)
                        new_val = EMPTY_PIXEL;
                    else
                        new_val=domain_val;
                
                    end
                end
                % range values should not need interpolation as x,y and origin are integers
                new_val = new_val + img2(range_point(2), range_point(1), :)*(1-alpha);
            elseif(in_domain)
                % if location is in our left image and not in right just take left
                % if point is between locations then interpolate
                new_val = interpolation_function(img1, domain_point);
                if(sum(new_val < 0, 'all')>0)
                   new_val = EMPTY_PIXEL; 
                end
            elseif(in_range)
                % if location is in our right image and not in left just take right
                % range values should not need interpolation as x,y and origin are integers
                new_val = img2(range_point(2), range_point(1), :);

            else
                % if location is in neither then set to black (0 0 0)
                % mark as not from either, will be mapped to zero
                new_val = EMPTY_PIXEL;
            end
            merged_img(y,x,:) = new_val;
        end
    end
end
