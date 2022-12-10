# Dogs and Cats 
#include((@__DIR__)*"/utils.jl")
using CUDA
using Metalhead
using Flux
using StatsBase
using ScikitLearn

# Define Model and adjoining function 
#nn_model = VGG19().layers[1:26]; #VGG is a good performer - 96%+ accuracy 
#nn_model = ResNet().layers[1:20]; #ResNet is a poor performer - 70% accuracy 
#nn_model = GoogleNet().layers[1:19] #GoogleNet gets 85% accuracy 
neurons = size(nn_model(rand(Float32, 224, 224, 3, 1)))[1]
#capture_bottleneck = create_bottleneck_pipeline(nn_model);


# create dataset for training and cross-validation 
cats = "train/".*StatsBase.sample(readdir("train/")[contains.(readdir("train/"), r"cat")], 1000, replace=false)
dogs = "train/".*StatsBase.sample(readdir("train/")[contains.(readdir("train/"), r"dog")], 1000, replace=false)
train_paths = [cats; dogs]

# create cat and dog features 
#dog_features = @time capture_bottleneck.(dogs);
#cat_features = @time capture_bottleneck.(cats);
dog_features = @time [nn_model.(Metalhead.preprocess.(load.(x))) for x in dogs] |> gpu
cat_features = @time [nn_model.(Metalhead.preprocess.(load.(x))) for x in cats] |> gpu 

# create a dataset for training 
allfeatures = zeros(Float32, 2000, neurons) |> CuArray;
[allfeatures[i, :] .= x for (i,x) in enumerate(vcat(dog_features, cat_features))];
allfeatures = Array(allfeatures);

# create ternary function that can be broadcast 
label_dog_cat(path) = contains(path, r"cat") ? "cat" : "dog";
y = label_dog_cat.([dogs; cats]);

# Run SVM with bottleneck as features 

## LOADING SCIKITLEARN 
@sk_import svm: LinearSVC
#@sk_import model_selection: RepeatedStratifiedKFold 
#@sk_import model_selection: cross_val_score
import ScikitLearn: CrossValidation
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#svm = LinearSVC(C=.01, loss="hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#svm.fit(allfeatures, y)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=3403)
out = cross_val_score(svm, allfeatures, y, cv = RSK.split(allfeatures,  y))

