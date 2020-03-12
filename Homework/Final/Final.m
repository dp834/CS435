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

%image_1_location = 'images/left.jpg';
image_1_location = 'images/im1.jpg';
left_img = imread(image_1_location);
% Images are rotated for some reason so just transpose them
%left_img = rot90(left_img, 3);

%image_2_location = 'images/right.jpg';
image_2_location = 'images/im2.jpg';
right_img = imread(image_2_location);
%right_img = rot90(right_img, 3);

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
stitched_image = stitch_images(left_img, right_img, source, target, @fast_interp);
%stitched_image = stitch_images(right_img, left_img, target, source, @fast_interp);

imshow(stitched_image);

%% Question 3
%

%% Functions
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
    xs = [xs(1), xs(2), xs(3), xs(4),  1, size(img2, 1),            1, size(img2,1)];
    ys = [ys(1), ys(2), ys(3), xs(4),  1,             1, size(img2,2), size(img2, 2)];

    [minx, minxi] = min(xs);
    [maxx, maxxi] = max(xs);
    [miny, minyi] = min(ys);
    [maxy, maxyi] = max(ys);

    new_size = [uint32(maxy-miny), uint32(maxx-minx), 3];

    if(minxi == minyi)
        origin = [-minx, -miny];
    elseif(minxi == maxyi)
        origin = [-minx, double(new_size(1)) - maxy];
    elseif(maxxi == minyi)
        origin = [double(new_size(2)) - maxx, -miny];
    elseif(maxxi == maxyi)
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
    new_img(:,:,:) = padarray(img, [9 9]);
    region = descriptor_func(y-4:y+4,x-4:x+4,:);
    descriptor_vector = reshape(region, 9*9*3, 1);
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
    output_img = zeros(size(img));
    octave_count = size(DoG, 1);
    scale_count = size(DoG, 2);

    for octave=1:octave_count
        scale_factor = 2^(octave-1);
        for scale=1:scale_count
            img = DoG{octave}{scale};
            for y=2:size(img, 1)-1
                for x=2:size(img, 2)-1

                   points_to_test = generate_potential_local_maxima(x, y, scale, octave, DoG);
                   maxima = max(points_to_test(:));

                   val = img(y,x);
                   if val > maxima
                       y_loc = y*scale_factor;
                       x_loc = x*scale_factor;
                       output_img(x_loc, y_loc) = 1;
                   end
                end
            end
        end
    end
end

function points = generate_potential_local_maxima(x, y, scale, octave, DoG)
    points = zeros(3,3,3);
    y = y+1;
    x = x+1;

    curr_DoG = DoG{octave}{scale};
    curr_DoG = padarray(curr_DoG, [1 1]);

    if scale > 1
        higher_DoG = DoG{octave}{scale-1};
        higher_DoG = padarray(higher_DoG, [1 1]);
        points(3,:,:) = higher_DoG(y-1:y+1,x-1:x+1);
    end

    if scale < size(DoG, 2)
        lower_DoG = DoG{octave}{scale+1};
        lower_DoG = padarray(lower_DoG, [1 1]);
        points(1,:,:) = lower_DoG(y-1:y+1,x-1:x+1);
    end

    points(2,1,:) = curr_DoG(y-1,x-1:x+1);
    points(2,2,1) = curr_DoG(y,x-1);
    points(2,2,2) = 0;
    points(2,2,3) = curr_DoG(y,x+1);
    points(2,3,:) = curr_DoG(y+1,x-1:x+1);
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
            DoG{octave}{scale} = scale_space{octave}{scale+1} - scale_space{octave}{scale};
        end
    end
end

% Computes Scale-Space Image Pyramid
% Input: Original Image, Number of Octaves, Number of Scales, Subsampling Factor
% Output: Cell Array of Images in Scale-Space Image Pyramid
function scale_space = compute_scale_space(im, octave_count, scale_count, reduction_factor)
    img = rgb2gray(im);
    scale_space = cell(4,5);
    for octave=1:octave_count
        sigma = 1.6;
        for scale=1:scale_count
            gauss_kernel = gaussian_filter(ceil(3*sigma), sigma);
            smoothed_gray = conv2(img, gauss_kernel, 'same');
            scale_space{octave}{scale} = uint8(smoothed_gray);
            sigma = sigma * sqrt(2);
        end
        img = img(1:reduction_factor:end, 1:reduction_factor:end, :);
    end
end
function scale_space = compute_scale_space2(im, octave_count, scale_count, reduction_factor)
    img = rgb2gray(im);
    scale_space = cell(4,5);
    sigma = 1.6;
    orig_img(:,:) = img(:,:);
    for scale=1:scale_count
        gauss_kernel = gaussian_filter(ceil(3*sigma), sigma);
        img = orig_img(:,:);
        for octave=1:octave_count
            smoothed_gray = conv2(img, gauss_kernel, 'same');
            scale_space{octave}{scale} = uint8(smoothed_gray);
            img = img(1:reduction_factor:end, 1:reduction_factor:end, :);
        end
        sigma = sigma * sqrt(2);
    end
end


function visualize_scale_space(scale_space)
    octaves = size(scale_space, 1);
    scales = size(scale_space, 2);
    figure;
    for y=1:octaves
        for x=1:scales
            subplot(octaves, scales, scales*(y-1) + x);
            imshow(scale_space{y}{x});
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
