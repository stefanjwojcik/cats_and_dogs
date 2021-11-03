### Transfer Learning with Julia

This is a tutorial about transfer learning with Julia. Along the way, we will also discuss important differences between Julia's Flux.jl system and popular alternatives like Keras. This tutorial draws heavily on the [dog cat tutorials](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for Keras. 

It's important to note outright that this tutorial is about how to fine-tune the upper layers of a pretrained network for a specific task. Instead, this tutorial uses the output of a lower layer of a pretrained network and uses it in a separate machine learning model - like a Support Vector Machine. 

The core idea is that the lower layers of a pretrained network capture deep abstract features about an image - such as distinct combinations of lines and points that might correspond to fur, wrinkles, eyes, noses, or structures. Neural networks trained on the ImageNet dataset - perhaps the most famous computer vision dataset - are trained with millions of images to classify a thousand common things. Imagenet includes things like trucks, elephants, balloons, and strawberries. 

## Prerequisites 

Let's load some key libraries. The most notable libraries here are Flux and Metalhead. Flux is Julia's deep learning library, and Metalhead is a library that provides access to pretrained deep learning models like ResNet for VGG19, which will be enormously useful to us. 

```julia
# Load the necessary libraries here 
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

@sk_import svm: LinearSVC
@sk_import model_selection: RepeatedStratifiedKFold 
@sk_import model_selection: cross_val_score


include((@__DIR__)*"/utils.jl")

```

We need to define some utilities here. One of the oddest things that I've found is that we still need to use a bit of existing Python code to preprocess images. Specifically, Tensorflow libraries use the means of the ImageNet data to rescale images prior to processing. Julia allows easy interoperability with Python, so below we'll call Python to load some preprocessing functions. If you get an error that these libraries are not installed, check out the [Tensorflow page](https://www.tensorflow.org/install/pip#system-install) for guidance on installation. 

Below, the `py_preprocess_input` command copies an image and executes the preprocessing function. We then call that function on an empty array to generate the values that are used to normalize each image with Python. That is, the 'preprocess_input' function in Python normalizes images based on the average pixel intensity of the ImageNet dataset. Replicating this process improves performance of the image processing. 

```julia

py"""
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
"""

function py_process_input(image_array)
    image_array_cx = deepcopy(image_array)
    image_array_cx .= py"preprocess_input"(image_array_cx)
    return image_array_cx
end

### JULIA FUNCTIONS (first draws on imagenet means )

imagenet_means = mean(py_process_input(zeros(224, 224, 3)), dims=(1, 2));
```


```julia
    # Function to return a function that scales appropriately. 
function image_net_gen_scale(imagenet_means)
    #imagenet_means = [103.93899999996464, 116.77900000007705, 123.67999999995286]
    function pyscale(x::AbstractArray)
        dx = copy(x)
        # swap R and G channels like python does - only during channels_last 
        dx[:, :, :, 1], dx[:, :, :, 3] = dx[:, :, :, 3], dx[:, :, :, 1]
        dx[:, :, :, 1] .+= imagenet_means[1]
        dx[:, :, :, 2] .+= imagenet_means[2]
        dx[:, :, :, 3] .+= imagenet_means[3]
        #return cor(collect(Iterators.flatten(dx)),collect(Iterators.flatten(py_scaled_image)))
        return(dx)
    end
end

# The actual function that will be used 
image_net_scale = image_net_gen_scale(imagenet_means)

```

We create a pipeline that will power the image processing of the dog and cat images. The @pipe macro allows us to pass the results of one function straight to the next via a series of anonymous functions. In series we load the image, resize to a square 224 x 224, convert to an array with the `channelview` function, then we swap the order of the dimensions, scale the image, and pass the result through the neural model. Notice that this function is a 

```julia
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

```


# Loading image paths 
```julia
# create dataset for training and cross-validation 
cats = "train/".*StatsBase.sample(readdir("train/")[contains.(readdir("train/"), r"cat")], 1000, replace=false)
dogs = "train/".*StatsBase.sample(readdir("train/")[contains.(readdir("train/"), r"dog")], 1000, replace=false)
train_paths = [cats; dogs]
```


```julia 
# The Model and adjoining function 
nn_model = VGG19().layers[1:25];
capture_bottleneck = create_bottleneck_pipeline(nn_model);

# create cat and dog features - see next chunk 
#dog_features = @time capture_bottleneck.(dogs);
#cat_features = @time capture_bottleneck.(cats);

function capture_dogs_cats(paths::Array{String, 1})
    allfeatures = zeros(Float32, length(paths), 4096);
    @showprogress for (i, x) in enumerate(paths)
        allfeatures[i, :] .= capture_bottleneck(x)
    end
    return allfeatures
end

# create a dataset for training 
dog_cat_features = capture_dogs_cats(vcat(dogs[1:100], cats[1:100]));

# create ternary function that can be broadcast 
label_dog_cat(path) = contains(path, r"cat") ? "cat" : "dog";
y = label_dog_cat.([dogs[1:100]; cats[1:100]]);

# SVM definition 
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

RSK = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=3403)
out = cross_val_score(svm, dog_cat_features, y, cv = RSK.split(dog_cat_features,  y))

```