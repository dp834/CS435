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

% image_1_location = 'images/left.jpg';
image_1_location = 'images/im1.jpg';
% image_1_location = 'images/test1.png';
left_img = imread(image_1_location);
% Images are rotated for some reason so just transpose them
% left_img = rot90(left_img, 3);

% image_2_location = 'images/right.jpg';
image_2_location = 'images/im2.jpg';
% image_2_location = 'images/test2.png';
right_img = imread(image_2_location);
% right_img = rot90(right_img, 3);

NUM_OCTAVES = 4;
NUM_SCALES = 5;
REDUCTION_FACTOR = 2;  % value is squared so images are subsampled to 1/4


% Question 3
% Compute Scale Space Pyramid
fprintf('Starting Scale Space Pyramid\n');
img1 = left_img(1:10:end, 1:10:end, :);
img2 = right_img(1:10:end, 1:10:end, :);


img1_scale_space = compute_scale_space(img1, NUM_OCTAVES, NUM_SCALES, REDUCTION_FACTOR);
img2_scale_space = compute_scale_space(img2, NUM_OCTAVES, NUM_SCALES, REDUCTION_FACTOR);
% visualize_scale_space(img1_scale_space);

% Question 4
% Find keypoints from local maxima of difference of gaussians
% Prune keypoints:
%   Eliminate extrema too close to boarder (based on kernel size of next part)
%   Find the standard deviation around each keypoint, filter those with a std < thresh due to low contrast
% Compute Local Maxima
fprintf('Starting DoG/maxima img1\n');

img1_DoG = compute_dog(img1_scale_space);
% img1_local_maxima = compute_local_maxima(img1, img1_DoG);
img1_local_maxima = imregionalmax(rgb2gray(img1));

fprintf('Starting DoG/maxima img2\n');
img2_DoG = compute_dog(img2_scale_space);
% img2_local_maxima = compute_local_maxima(img2, img2_DoG);
img2_local_maxima = imregionalmax(rgb2gray(img2));



fprintf('Starting pruning, rest is fast\n');
img2_key_points = prune_unstable_maxima(img2, img2_local_maxima, 70);
img1_key_points = prune_unstable_maxima(img1, img1_local_maxima, 70);





%%

img1_feature_vectors = get_features_from_key_points(img1, img1_key_points);
img2_feature_vectors = get_features_from_key_points(img2, img2_key_points);
key_point_correspondences = [];

fprintf("Starting keypoint matching\n");
img1_pairs = zeros(size(img1_feature_vectors, 2),2);
for i = 1:size(img1_pairs,1)
    img1_pairs(i,:) = [i sort_by_similarity(img1_feature_vectors(:,i), img2_feature_vectors)];
end

img2_pairs = zeros(size(img2_feature_vectors, 2),2);
for i = 1:size(img2_pairs,1)
    img2_pairs(i,:) = [sort_by_similarity(img2_feature_vectors(:,i), img1_feature_vectors) i];
    if(img1_pairs(img2_pairs(i,2),1) == i)
        key_point_correspondences(end+1, :) = [img2_pairs(i,1), i];
    end
end

% Draw both images and mark the key point correspondences
img_out = [img1 img2];
imshow(img_out);
img1_key_point_correspondences = [];
img2_key_point_correspondences = [];
for i = 1:size(key_point_correspondences,1)
    if abs(img1_key_points(key_point_correspondences(i,1),2) - img2_key_points(key_point_correspondences(i,2),2)) < .05*size(img1, 1)
        img1_key_point_correspondences(end+1, :) = img1_key_points(key_point_correspondences(i,1),:);
        img2_key_point_correspondences(end+1, :) = img2_key_points(key_point_correspondences(i,2),:);
        draw_line(img1_key_point_correspondences(end,:), img2_key_point_correspondences(end,:), size(img1,2));
    end
end
saveas(gcf, strcat('key_point_correspondences.png'));



%%
 

fprintf("Starting RANSAC\n");
[img1_points, img2_points] = ransac(img1_key_point_correspondences, img2_key_point_correspondences, 160000);

%%
imshow([img1 img2])
for i=1:size(img1_points)
    draw_circle(img1_points(i,1), img1_points(i,2), 3, 'r', 1);
end
for i=1:size(img2_points)
    draw_circle(img2_points(i,1)+size(img1,2), img2_points(i,2), 3, 'r', 1);
end

for i=1:size(img1_points)
   draw_line(img1_points(i,:), img2_points(i,:), size(img1,2)); 
end

%%
stitched_image = stitch_images(img1, img2, img2_points, img1_points, @fast_interp);

imshow(stitched_image);



% Functions











%%%% Feature Vector Computation

function features = get_features_from_key_points(img, key_points)
    patch_size = 15;
    half_size = floor(patch_size/2);

    xs = key_points(:,1);
    ys = key_points(:,2);
    img = padarray(img, [patch_size patch_size]);
    features = zeros(patch_size*patch_size*3, numel(xs));
    for i = 1:numel(xs)
        x = xs(i)+patch_size;
        y = ys(i)+patch_size;
        
        fv = img(y-half_size:y+half_size,x-half_size:x+half_size,:);
        fv = reshape(fv, numel(fv),1);
        
        features(1:end, i) = fv;
    end
end

% given an array of features, and a feature to compare to will return a ordered list based on similarity
function most_similar = sort_by_similarity(source_fv, tests_fv)
    % index into original list, similarity
    similarities = zeros(size(tests_fv,2), 1);
    for item = 1:size(tests_fv,2)
        similarities(item) = compute_similarity(tests_fv(:,item), source_fv);
    end
    [~, most_similar] = min(similarities);

end

function similarity = compute_similarity(x, y)
    similarity = 0;
    for bin = 1:size(x,1)
        similarity = similarity + abs(x(bin) - y(bin));
    end
end






%%%% Ransac

function [kp_source_p, kp_target_p]= ransac(kp_source, kp_target, iter_count)
    
    
	for iter=1:iter_count
        min_dist = exp(1000);
        random_source = zeros(4,2);
        random_target = zeros(4,2);
        random_inds = randperm(size(kp_source, 1),4);    
        for i=1:4
            random_source(i,:) = kp_source(random_inds(i),:);
            random_target(i,:) = kp_target(random_inds(i),:);
        end    

        h_mat = compute_transformation_matrix(random_source, random_target);
        kp_projected = map_multiple_points(kp_source, h_mat);
        
        euclid_dist = sum(abs(kp_projected-kp_target));
        if euclid_dist< min_dist
            min_inds = random_inds;
            min_dist = euclid_dist;
        end
    end
    kp_source_p = kp_source(min_inds,:)
    kp_target_p = kp_target(min_inds,:)
    h_mat = compute_transformation_matrix(random_source, random_target);
    kp_projected_p = map_multiple_points(kp_source, h_mat);
    kp_projected_p(min_inds, :)
end




%%%% Extrema

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
                        output_img((y-1)*(scale_factor-1)+1, (x-1)*(scale_factor-1)+1) = 1;
                    end
                end
            end
        end

    end
end

function key_points = prune_unstable_maxima(img, candidates, threshold)
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


  function extremas = remove_unstable_extremas(img, extremas_all, border_dist, patch_dim, threshold)
    img = rgb2gray(img);
     [h, w] = size(img);
     extremas = extremas_all - edge(img); %sets edges in original to 0 (false)
     [i,j] = find(extremas==1); %only use indices of points that are identified as extrema 
     sz = floor(patch_dim/2); 
     for k=1:length(i)
         y = i(k);
         x = j(k);
         if (~is_too_close_to_border(x, y, border_dist))
             patch = img(y-sz:y+sz, x-sz:x+sz);
             if std2(patch) < threshold
                 extremas(y,x) = 0;
             end 
         else
             extremas(y,x) = 0;
         end 
     end

    function is_to_close = is_too_close_to_border(x, y, dist)
        is_to_close = (y < dist || y > (h-dist) || x < dist || x > (w-dist));
    end 
end







%%%% Stitch Images 

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





%%%% Transformation and Projection points

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

function output = map_point(H, p)
    output = H*p;
    output = [output(1)/output(3), output(2)/output(3)];
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





%%%% Gaussian Scale Space

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

function scale_space = compute_scale_space(im, octave_count, scale_count, reduction_factor)
    img = rgb2gray(im);
    scale_space = cell(4,5);
    for octave=1:octave_count
        sigma = 1.6;
        for scale=1:scale_count
%             gauss_kernel = gaussian_filter(ceil(3*sigma), sigma);
            gauss_kernel = fspecial('gaussian', ceil(3*sigma), sigma);
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




%%%% Drawing functions

function draw_line(coord1, coord2, width)
    hold on
    line([coord1(1) (coord2(1)+width)], [coord1(2) coord2(2)],'Color', 'red');
    hold off
    axis equal
end

function draw_circle(x,y,radius,color,fill_it)
    hold on
    th = linspace(0,2*pi,100);
    if fill_it
        fill(x + radius*cos(th), y + radius*sin(th), color, 'LineStyle', 'none', 'EdgeColor', 'none');
    else
        plot(x + radius*cos(th), y + radius*sin(th));
    end
    hold off
    axis equal
end
