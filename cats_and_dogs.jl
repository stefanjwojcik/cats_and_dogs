# Dogs and Cats 
using ScikitLearn
using Metalhead
using Flux
using Images
using ImageTransformations
using CUDA
using PyCall
using Pipe
using ProgressMeter
using Statistics 
using StatsBase
CUDA.allowscalar(false)

include((@__DIR__)*"/utils.jl")

"""
Before we can do any image modeling, we need to create a pipeline to process each image. In this pipeline, we will load the image into memory, resize it, reshape it to fit in our model, and finally generate features from it. 
"""
function create_bottleneck_pipeline(neural_model)
    function capture_bottleneck(image_path)
        out = @pipe load(image_path) |> #
        x -> imresize(x, 224, 224) |> #
        x -> channelview(x) * 255 |> #
        x -> permutedims(x, [2, 3, 1]) |> #
        x -> reshape(x, (1, 224, 224, 3) ) |> # Python style for comparison sake 
        x -> image_net_scale(x) |>
        x -> reshape(x, (224, 224, 3, 1)) |>
        x -> cflat(neural_model(x))
        return out
    end
end

# create dataset for training and cross-validation 
cats = "train/".*StatsBase.sample(readdir("train/")[contains.(readdir("train/"), r"cat")], 1000, replace=false)
dogs = "train/".*StatsBase.sample(readdir("train/")[contains.(readdir("train/"), r"dog")], 1000, replace=false)
train_paths = [cats; dogs]

# The Model and adjoining function 
nn_model = VGG19().layers[1:25];
capture_bottleneck = create_bottleneck_pipeline(nn_model);

# create cat and dog features 
dog_features = @time capture_bottleneck.(dogs);
cat_features = @time capture_bottleneck.(cats);

# create a dataset for training 
allfeatures = zeros(Float32, 2000, 4096);
[allfeatures[i, :] .= x for (i,x) in enumerate(vcat(dog_features, cat_features))];

# create ternary function that can be broadcast 
label_dog_cat(path) = contains(path, r"cat") ? "cat" : "dog";
y = label_dog_cat.([dogs; cats]);

# Run SVM with bottleneck as features 

## LOADING SCIKITLEARN 
@sk_import svm: LinearSVC
@sk_import model_selection: RepeatedStratifiedKFold 
#@sk_import model_selection: cross_val_score
import ScikitLearn: CrossValidation
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#svm = LinearSVC(C=.01, loss="hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#svm.fit(allfeatures, y)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=3403)
out = cross_val_score(svm, allfeatures, y, cv = RSK.split(allfeatures,  y))

################ SAME PIPELINE BUT USING THE NN MODEL FROM PYTHON 

function create_bottleneck_pipeline_python(neural_model)
    function capture_bottleneck(image_path)
        out = @pipe load(image_path) |> #
        x -> imresize(x, 224, 224) |> #
        x -> channelview(x) * 255 |> #
        x -> permutedims(x, [2, 3, 1]) |> #
        x -> reshape(x, (1, 224, 224, 3) ) |> # Python style for comparison sake 
        x -> image_net_scale(x) |>
        x -> cflat(neural_model(x))
        return out
    end
end

capture_bottleneck_py = create_bottleneck_pipeline_python(py"nn_model".predict)

# create cat and dog features 
dog_features = @time capture_bottleneck_py.(dogs);
cat_features = @time capture_bottleneck_py.(cats);

# create final dataset for training
allfeatures = zeros(Float32, 2000, 4096);
[allfeatures[i, :] .= @inbounds x for (i,x) in enumerate(vcat(dog_features, cat_features))];

# create ternary function that can be broadcast 
label_dog_cat(path) = contains(path, r"cat") ? "cat" : "dog";
y = label_dog_cat.([dogs; cats]);

# Run SVM with bottleneck as features 
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#svm = LinearSVC(C=.01, loss="hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#svm.fit(allfeatures, y)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=3403)
out = cross_val_score(svm, allfeatures, y, cv = RSK.split(allfeatures,  y))

# Test completely out of sample: 
randn(["train/".*readdir("train/") .∉ train_paths], 10)

trainpath_logical = in(train_paths).("train/".*readdir("train/"));
testpets = StatsBase.sample(readdir("train/")[.!trainpath_logical], 1000);

testfeatures = capture_bottleneck_py.("train".*testpets);
testY = label_dog_cat.("train/".*testpets);

svm.fit(allfeatures, testY)

# RANDOM THOUGHTS ABOUT BOOK **********

# Dr Seuss themed 

# Generate random sneech names 
sneech_names = ["Barnaby", "Bullabus", "Sycamore", "S"]

star_off_machine() = "removed stars"
star_on_machine() = "added stars"