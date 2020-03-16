%{
    Damien Prieur
    Partner: Daniel Schwartz
    CS 435
    Final
%}

%% Global Setup

format short

output_location_prefix = 'images/generated/';
%cleanup any previously generated images
delete('images/generated/*');
fprintf('Cleaned "images/generated"\n');

image_1_location = 'images/im1.jpg';
left_img = imread(image_1_location);

image_2_location = 'images/im2.jpg';
right_img = imread(image_2_location);

%% User-Defined Constants
NUM_OCTAVES = 4;
NUM_SCALES = 5;
REDUCTION_FACTOR = 2;  % value is squared so images are subsampled to 1/4

%% Question 1

% Points for                            left     |    right
% Webcam top right of left text     (1651, 1239) | (481, 1306)
% Bottom left of L in LG            (1647, 2190) | (595, 2269)
% Bottom right of the V key ink     (1479, 2616) | (295, 2719)
% Coke rop right of white barcode   (1801, 2994) | (421, 3124)
%{
x     = [1651 1647 1479 1801];
x_hat = [ 481  595  295  421];
y     = [1239 2190 2616 2994];
y_hat = [1306 2269 2719 3124];
%}



%% Question 2
source = [1710, 1237; ...
          1672, 3103; ...
          1456, 1145; ...
          2805, 3304];



target = [ 546, 1228; ...
           553, 3173; ...
           264, 1117; ...
          1652, 3208];

%stitched_image = stitch_images(left_img, right_img, source, target, @linear_interpolation);
%stitched_image = stitch_images(right_img, left_img, target, source, @linear_interpolation);
%stitched_image = stitch_images(left_img, right_img, source, target, @fast_interp);
%stitched_image = stitch_images(right_img, left_img, target, source, @fast_interp);

%imshow(stitched_image);

%% Question 3
% Compute Scale Space Pyramid
fprintf('Starting Scale Space Pyramid\n');
img1 = left_img;
img2 = right_img;
%


img1_scale_space = compute_scale_space(img1, NUM_OCTAVES, NUM_SCALES, REDUCTION_FACTOR);
img2_scale_space = compute_scale_space(img2, NUM_OCTAVES, NUM_SCALES, REDUCTION_FACTOR);

%% Question 4
% Find keypoints from local maxima of difference of gaussians
% Prune keypoints:
%   Eliminate extrema too close to boarder (based on kernel size of next part)
%   Find the standard deviation around each keypoint, filter those with a std < thresh due to low contrast
% Compute Local Maxima
fprintf('Starting DoG/maxima img1\n');

img1_DoG = compute_dog(img1_scale_space);
img1_local_maxima = compute_local_maxima(img1, img1_DoG);

fprintf('Starting DoG/maxima img2\n');
img2_DoG = compute_dog(img2_scale_space);
img2_local_maxima = compute_local_maxima(img2, img2_DoG);

%%
fprintf('Starting pruning, rest is fast\n');


%img1_local_maxima = imregionalmax(rgb2gray(img1));
%img2_local_maxima = imregionalmax(rgb2gray(img2));

img2_key_points = prune_unstable_maxima(img2, img2_local_maxima);
img1_key_points = prune_unstable_maxima(img1, img1_local_maxima);

% Question 5
% Find matching keypoints
% For each keypoint compute a feature vector and look for a match in the other set of keypoints

img1_feature_vectors = get_features_from_key_points(img1, img1_key_points);
img2_feature_vectors = get_features_from_key_points(img2, img2_key_points);

% keypoint matching

key_point_correspondences = [];

img1_pairs = zeros(size(img1_feature_vectors, 2),2);

for i = 1:size(img1_pairs,1)
    img1_pairs(i,:) = [i sort_by_similarity(img1_feature_vectors(:,i), img2_feature_vectors)];
end

img2_pairs = zeros(size(img2_feature_vectors, 2),2);

for i = 1:size(img2_pairs,1)
    img2_pairs(i,:) = [sort_by_similarity(img2_feature_vectors(:,i), img1_feature_vectors) i];
    if(img1_pairs(img2_pairs(i,1),2) == i)
        key_point_correspondences(end+1, :) = [img2_pairs(i,1), i];
    end
end

%% Draw both images and mark the key point correspondences
%img_out = [img1 img2];
%montage({img1,img2});
img1_key_point_correspondences = [];
img2_key_point_correspondences = [];
for i = 1:size(key_point_correspondences,1)
    if abs(img1_key_points(key_point_correspondences(i,1),2) - img2_key_points(key_point_correspondences(i,2),2)) < .10*size(img1, 1)
        if (size(img1,2) - img1_key_points(key_point_correspondences(i,1),1) + img2_key_points(key_point_correspondences(i,2),1)) < (size(img1,2) + size(img2, 2))/2
            img1_key_point_correspondences(end+1, :) = img1_key_points(key_point_correspondences(i,1),:);
            img2_key_point_correspondences(end+1, :) = img2_key_points(key_point_correspondences(i,2),:);
            %draw_line(img1_key_point_correspondences(end,:), img2_key_point_correspondences(end,:), size(img1,2));
        end
    end
end
%saveas(gcf, strcat(output_location_prefix, 'key_point_correspondences.png'));

%%

%[img1_points, img2_points] = ransac(img1_key_point_correspondences, img2_key_point_correspondences, log(1-.9)/log(1-.05^4));
stitched_image = stitch_images(img1, img2, img1_points, img2_points, @fast_interp);

imshow(stitched_image);

% Functions
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

    % one of the 4 outer corners is going to be mapped to a corner
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
                new_val = interpolation_function(img1, domain_point)/2;
                % range values should not need interpolation as x,y and origin are integers
                new_val = new_val + img2(range_point(2), range_point(1), :)/2;
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

function stitched_img = stitch_images(img1, img2, keypoint_src, keypoint_target, interpolation_func)
    H = compute_transformation_matrix(keypoint_src, keypoint_target);

    [new_image_size, origin] = get_combined_size(img1, img2, H);

    stitched_img = merge_images(img1, img2, H, new_image_size, origin, interpolation_func);
end

function output = map_point(H, p)
    output = H*p;
    output = [output(1)/output(3), output(2)/output(3)];
end

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

% Computes Descriptor Vector
% Input: Image and location (x,y) of keypoint
% Output: Descriptor Vector
function descriptor_vector = compute_descriptor(img, x, y, descriptor_func)
    descriptor_vector = descriptor_func(img, x, y);
end


% Computes similarity between two representations of images
% Input: Two descriptor vectors to compare for similarity
% Output: Similarity measure for two descriptor vector
function summation = compute_similarity(a, b)
    summation = 0;
    for j=1:size(a,2)
        summation = summation + min(a(j), b(j));
    end
end


% Computes Local Maxima of a DoG
% Input: Cell Array containg DoG across different octaves
% Output: Binary matrix of local maxima
function output_img = compute_local_maxima(img, DoG)
    output_img = zeros(size(img,1),size(img,2));
    octave_count = size(DoG, 1);
    scale_count = size(DoG, 2);

    for octave=1:octave_count
        for scale=1:scale_count
            DoG{octave, scale} = padarray(DoG{octave, scale}, [1, 1]);
        end
    end

    for octave=1:octave_count
        scale_factor = 2^(octave-1);
        DoG_octave = zeros(size(DoG{octave, 1}, 1), size(DoG{octave, 1},2), 2 + scale_count);

        for scale=1:scale_count
            DoG_octave(:,:,scale+1) = DoG{octave, scale};
        end


        for x = 2:size(DoG_octave,2)-1
            for y = 2:size(DoG_octave,1)-1
                for z = 2:size(DoG_octave,3)-1
                    % Look at the cube around the point and if it's the max then we mark it as a max
                    if(3*3+5 == find(DoG_octave(y-1:y+1,x-1:x+1,z-1:z+1) == max(DoG_octave(y-1:y+1,x-1:x+1,z-1:z+1), [], 'all')))
                        output_img((y-1)*scale_factor, (x-1)*scale_factor) = 1;
                    end
                end
            end
        end

    end
end

% Computes Difference of Gaussian in Image Pyramid
% Input: Cell Array of Images in Scale-Space Image Pyramid
% Output: Difference of Gaussians across different Octaves in Cell Array
function DoG = compute_dog(scale_space)
    octave_count = size(scale_space, 1);
    scale_count = size(scale_space, 2);
    DoG = cell(octave_count,scale_count-1);
    for octave=1:octave_count
        for scale=1:scale_count-1
            DoG{octave, scale} = scale_space{octave, scale+1} - scale_space{octave, scale};
        end
    end
end

% Computes Scale-Space Image Pyramid
% Input: Original Image, Number of Octaves, Number of Scales, Subsampling Factor
% Output: Cell Array of Images in Scale-Space Image Pyramid
function scale_space = compute_scale_space(im, octave_count, scale_count, reduction_factor)
    img = rgb2gray(im);
    scale_space = cell(4,5);
    sigma_o = 1.6;
    for octave=1:octave_count
        for scale=1:scale_count
            sigma = sigma_o*2^(octave-1)*sqrt(2)^(scale-1);
            gauss_kernel = gaussian_filter(ceil(3*sigma), sigma);
            smoothed_gray = conv2(img, gauss_kernel, 'same');
            scale_space{octave, scale} = uint8(smoothed_gray);
            sigma = sigma * sqrt(2);
        end
        img = img(1:reduction_factor:end, 1:reduction_factor:end, :);
    end
end

function visualize_scale_space(scale_space)
    octaves = size(scale_space, 1);
    scales = size(scale_space, 2);
    figure;
    for y=1:octaves
        for x=1:scales
            subplot(octaves, scales, scales*(y-1) + x);
            imshow(scale_space{y, x});
            axis on;
        end
    end
    saveas(gcf, 'scale_space.png');
end

% Computes Gaussian Filter
function f=gaussian_filter(n,s)
    x = -1/2:1/(n-1):1/2;
    [Y,X] = meshgrid(x,x);
    f = exp( -(X.^2+Y.^2)/(2*s^2) );
    f = f / sum(f(:));
end

function key_points = prune_unstable_maxima(img, candidates)
    % Compute Edge Pixels
    edge_points = edge(rgb2gray(img));

    % The patch will go this count in all directions to make the patch
    % So a WIDTH of 4 will be a 9x9 patch
    WIDTH = 4;
    % Since i will be adding each color channel divide by 3
    CONTRAST_THRESH = 70*3;

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


function fv = rgb_fv(img, x, y)
    fv = img(y-4:y+4,x-4:x+4,:);
    fv = reshape(fv, numel(fv),1);
end

function features = get_features_from_key_points(img, key_points)
    xs = key_points(:,1);
    ys = key_points(:,2);
    %img = padarray(img, [9 9]);
    features = zeros(9*9*3, numel(xs));
    for i = 1:numel(xs)
        features(:, i) = reshape(img(ys(i)-4:ys(i)+4, xs(i)-4:xs(i)+4, :), [9*9*3, 1]);
    end
end

% given an array of features, a feature to compare to, and similarity_func will return a ordered list based on similarity
function most_similar = sort_by_similarity(source_fv, tests_fv)
    % index into original list, similarity
    similarities = zeros(size(tests_fv,2), 1);
    for item = 1:size(tests_fv,2)
        similarities(item) = histogram_similarity(tests_fv(:,item), source_fv);
    end
    [~, most_similar] = min(similarities);

end

% compute similarity of two histogram feature vectors
function similarity = histogram_similarity(hist_a, hist_b)
    similarity = 0;
    for bin = 1:size(hist_a,1)
        similarity = similarity + abs(hist_a(bin) - hist_b(bin));
    end
end

function [kp_source_p, kp_target_p]= ransac(kp_source, kp_target, iter_count)
    min_dist = exp(1000);

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

function draw_line(coord1, coord2, width)
    hold on
    line([coord1(1) (coord2(1)+width)], [coord1(2) coord2(2)],'Color', 'red');
    hold off
    axis equal
end
