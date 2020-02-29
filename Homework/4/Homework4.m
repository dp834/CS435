%{
    Damien Prieur
    CS 435
    Assignment 4
%}

%% Question 1
% Compute the SVD of the matrix to find the minimum weight cuts of the graph given

D =  [ (exp(-2) + exp(-5) + exp(-11)) 0 0 0; ...
        0 (exp(-2) + exp(-3) + exp(-5)) 0 0; ...
        0 0 (exp(-5) + exp(-3) + exp(-2)) 0; ...
        0 0 0 (exp(-11) + exp(-5) + exp(-2));];
W =  [0 exp(-2)  exp(-5)  exp(-11); ...
        exp(-2)  0 exp(-3) exp(-5); ...
        exp(-5)  exp(-3) 0 exp(-2); ...
        exp(-11) exp(-5) exp(-2) 0;];


fprintf('\n\n-----------------------------------------\n');
fprintf('SVD\n');
[U, S, V] = svd(D-W)
fprintf('-----------------------------------------\n');

%% Question 2
% For each image in the dataset:
%   compute the grayscale histogram with 256 bins
%   extract a class label from the first three characters of the filename (neg, pos) use enumeration (neg=0, pos=1)
%
% Make training sets (2/3 training, 1/3 testing)
% Then classify the testing data using k-nearest neighbors where k = 5
%   Use: sim(a,b) = sum_{j=1}^{D}min(a_j,b_j)

fprintf('\n\n-----------------------------------------\n');
fprintf('Histogram\n');
kNN(@histogram_fv, @histogram_similarity, 5);
fprintf('-----------------------------------------\n');

%% Question 3
% Divide graysscale into 10 non-overlapping 20x20 sub-images, computing an 8-bin HOG at each sub-image.
% Concatenate the ten 8-bin HOGs to form an 80-feature representation of the image.
% Repeat k-nearest-neighbors as in part 1

fprintf('\n\n-----------------------------------------\n');
fprintf('HOG\n');
kNN(@hog_fv, @histogram_similarity, 5);
fprintf('-----------------------------------------\n');

%% functions
% Used for both questions, wraps the similar functionality of the two parts into one. Just pass a new feature vector function and specify k
function none = kNN(fv_func, similarity_func,  k)
    % Taken from assignment
    d = 'CarData/TrainImages';
    files = dir(d);
    X = [];
    Y = [];
    fnames = strings();
    for f = files'
        if ~f.isdir
            img = imread([d, '/', f.name]);
            %do whatever you need to in order to generate feature vector for im
            %which I’ll call fv
            X(end+1,:) = fv_func(img);
            Y(end+1,1) = ~strcmp(f.name(1:3),'neg');
            fnames(size(Y,1),1) = f.name;
        end
    end

    rng(0);
    inds = randperm(size(X,1));
    num = size(X,1)/3;
    X = X(inds,:);
    Y = Y(inds,:);
    fnames = fnames(inds,:);

    Xtrain = X(1:2*num,:);
    Ytrain = Y(1:2*num,:);
    fnames_train = fnames(1:2*num,:);

    Xtest = X(2*num+1:end,:);
    Ytest = Y(2*num+1:end,:);
    fnames_test = fnames(2*num+1:end,:);

    guess_test = zeros(size(Ytest));

    asdf = 0;

    for im = 1:size(Xtest,1)
        similarities = sort_by_similarity(Xtrain, Xtest(im,:), similarity_func);
        for i = 1:k
            guess_test(im) = guess_test(im) + Ytrain(similarities(i));
        end
        guess_test(im) = round(guess_test(im)/k);
    end

    % compute results
    res_diff = Ytest - guess_test;

    % Just printing information for the user
    correctly_identified = sum(res_diff== 0) / numel(res_diff);
    fprintf('Correctly identified:  %f %%\n', correctly_identified);
    % Find example of correctly true and correctly false
    [index, ~] = find(Ytest==1 & guess_test==1);
    if(size(index,1) == 0)
        fprintf('Correctly positive: None\n');
    else
        fprintf('Correctly positive: %s\n', fnames_test(index(1)));
    end
    [index, ~] = find(Ytest==0 & guess_test==0);
    if(size(index,1) == 0)
        fprintf('Correctly negative: None\n');
    else
        fprintf('Correctly negative: %s\n', fnames_test(index(1)));
    end

    incorrect_false_pos  = sum(res_diff==-1) / numel(res_diff);
    fprintf('Incorrectly identified as a car:  %f %%\n', incorrect_false_pos);
    % Find example
    [index, ~] = find(Ytest==0 & guess_test==1);
    if(size(index,1) == 0)
        fprintf('Incorrectly identified as a car: None\n');
    else
        fprintf('Incorrectly identified as a car: %s\n', fnames_test(index(1)));
    end

    incorrect_false_neg  = sum(res_diff== 1) / numel(res_diff);
    fprintf('Incorrectly identified as not a car:  %f %%\n', incorrect_false_neg);
    % Find example
    [index, ~] = find(Ytest==1 & guess_test==0);
    if(size(index,1) == 0)
        fprintf('Incorrectly identified as not a car: None\n');
    else
        fprintf('Incorrectly identified as not a car: %s\n', fnames_test(index(1)));
    end


end

% given an array of features, a feature to compare to, and similarity_func will return a ordered list based on similarity
function most_similar = sort_by_similarity(fv_compare, fv_test, similarity_func)
    % index into original list, similarity
    similarities = zeros(size(fv_compare,1), 1);
    for item = 1:size(fv_compare,1)
        similarities(item) = similarity_func(fv_compare(item,:), fv_test);
    end

    [~, most_similar] = sortrows(similarities,1,'descend');

end

% generate histogram feature vector
function fv = histogram_fv(img)
    fv = zeros(256,1);
    for bin = 1:256
        fv(bin) = sum(img==bin, 'all');
    end
    fv = fv/numel(img);
end

% compute similarity of two histogram feature vectors
function similarity = histogram_similarity(hist_a, hist_b)
    similarity = 0;
    for bin = 1:size(hist_a,2)
        similarity = similarity + min(hist_a(bin), hist_b(bin));
    end
end


% generate hog feature vector
function fv = hog_fv(img)
    % Not doing chunking dynamically, we know that the images are 40x100 and we want 20x20 chunks
    % This leads us to 2x5 chunks

    chunk_width  = 20;
    chunk_height = 20;
    x_chunks = 5;
    y_chunks = 2;
    gradient_bins = 8;

    fv = zeros(x_chunks*y_chunks*gradient_bins, 1);

    for i = 1:y_chunks
        for j = 1:x_chunks
            offset_in = (i-1) * gradient_bins * x_chunks + (j-1) * gradient_bins + j;
            % A bit complex of a statement but it's just going through each chunk and getting the hog for each and putting it into the feature vector
            fv(offset_in:offset_in + gradient_bins - 1) = ...
                            compute_gradient_bins(img((i-1)*chunk_height + 1:i*chunk_height, (j-1)*chunk_width + 1: j*chunk_width), gradient_bins);
        end
    end
end

% Compute the gradient and place them into bins
function bins = compute_gradient_bins(img, bin_count)

    % compute gradients
    partial_x_convolution = [
    1/3 0 -1/3 ;
    1/3 0 -1/3 ;
    1/3 0 -1/3 ; ];

    partial_y_convolution = [
     1/3  1/3  1/3 ;
      0    0    0  ;
    -1/3 -1/3 -1/3 ; ];

    partial_x_img = conv2(img, partial_x_convolution, 'same');
    partial_y_img = conv2(img, partial_y_convolution, 'same');

    partial_x_img = reshape(partial_x_img, [1 numel(partial_x_img)]);
    partial_y_img = reshape(partial_y_img, [1 numel(partial_y_img)]);

    % place gradients into bins
    bins = zeros(bin_count,1);

    bin_width = 2*pi/bin_count;

    for i = 1:size(partial_x_img,2)
        % if no gradient information skip
        if(partial_y_img(i) == 0 && partial_x_img(i) == 0)
            continue
        end
        val = atan2(-partial_y_img(i), partial_x_img(i));

        % atan2 returns values from [-pi, pi] but i want [0, 2pi]
        % values that are negative are really just 2pi away
        if(val < 0)
            val = 2*pi + val;
        end

        % can't have zero so force it to map to the first bin
        if(val == 0)
            val = bin_width;
        end

        bins(ceil(val/bin_width)) = bins(ceil(val/bin_width)) + 1;
    end
end
