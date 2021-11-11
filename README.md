### SVM Transfer Learning with Julia

This is a tutorial about transfer learning with Julia. Along the way, we will also discuss important differences between Julia's Flux.jl system and popular alternatives like Keras. This tutorial draws heavily on the [dog cat tutorials](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for Keras. 

The core idea of transfer learning is that the lower layers of a pretrained network capture abstract features about an image - such as distinct combinations of lines and points that might correspond to fur, wrinkles, eyes, noses, or other structures. The top layer output of a pretrained neural network is a low-dimensional representation of all of these abstract features for a given image - similar to an [embedding](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture). 

Neural networks trained on the ImageNet dataset - perhaps the most famous computer vision dataset - are trained with millions of images to classify a thousand common things. Imagenet includes things like trucks, elephants, balloons, and strawberries. This kind of data provides a huge amount of visual information that can be captured by the neural network model. Therefore, there are a huge number of abstract features represented which we can leverage rather cheaply. 

You will see  [tutorials](https://fluxml.ai/tutorials/2020/10/18/transfer-learning.html) that retrain the upper layers of a pretrained neural network for a specific task. We will use a different but related strategy for modeling - we will extract the output of the penultimate layer of a pretrained deep neural network and uses itmany in a separate machine learning model - in this case a Support Vector Machine. We do this for two reasons. First, this method doesn't require a GPU (although having one will make some parts go faster). Second, it's slightly faster to extract the penultimate layer and pass it to an SVM than to retrain the upper layers. 


## Prerequisites 

Let's start julia and load some key libraries. I am on Julia 1.5.2, but I suspect newer versions would work just fine. For an editor, I am using [VSCODE](https://code.visualstudio.com/) with the [Julia extension](https://code.visualstudio.com/docs/languages/julia). 

Let's start Julia in a specific project environment `using Pkg; Pkg.activate("cats_and_dogs")` so we can keep track of all our dependencies. 

The most notable libraries I'm going to load are Flux and Metalhead. Flux is Julia's deep learning library, and Metalhead is a library that provides access to pretrained deep learning models like ResNet for VGG19, which will be enormously useful to us. 

```julia
# Load the necessary libraries here 
using ScikitLearn
using Metalhead
using Flux
using Images
using ImageTransformations
using CUDA
using PyCall
using Random # for rng 
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

It will be useful to define some utilities here. I'm going to use a tiny bit of code inspired by the `preprocess_input` function from tensorflow to preprocess images. It will modify them so they are normalized to the average pixel in the ImageNet source database. 

Below, the `py_preprocess_input` command copies an image and executes the preprocessing function. We then call that function on an empty array to generate the values that are used to normalize each image with Python. That is, the 'preprocess_input' function in Python normalizes images based on the average pixel intensity of the ImageNet dataset. Replicating this process improves performance of the image processing. 

```julia
function jimage_net_scale(dx::AbstractArray)
    imagenet_means = [-103.93899999996464, -116.77900000007705, -123.67999999995286]
    #dx = copy(x)
    # swap R and G channels like python does - only during channels_last 
    dx[:, :, :, 1], dx[:, :, :, 3] = dx[:, :, :, 3], dx[:, :, :, 1]
    dx[:, :, :, 1] .+= imagenet_means[1]
    dx[:, :, :, 2] .+= imagenet_means[2]
    dx[:, :, :, 3] .+= imagenet_means[3]
    #return cor(collect(Iterators.flatten(dx)),collect(Iterators.flatten(py_scaled_image)))
    return(dx)
end
```

We now create a pipeline that will power the image processing of the dog and cat images. The @pipe macro allows us to pass the results of one function straight to the next via a series of anonymous functions. In series we load the image, resize to a square 224 x 224, convert to an array with the `channelview` function, then we swap the order of the dimensions, scale the image, and pass the result through the neural model. The 224 resizing is so that the image is of a compatible size with the neural network model we're going to use. 

```julia
function create_bottleneck_pipeline(neural_model)
    function capture_bottleneck(image_path)
        out = @pipe load(image_path) |> #
        x -> imresize(x, 224, 224) |> #
        x -> channelview(x) * 255 |> #
        x -> permutedims(x, [2, 3, 1]) |> #
        x -> reshape(x, (1, 224, 224, 3) ) |> # Python style for comparison sake 
        x -> jimage_net_scale(x) |>
        x -> reshape(x, (224, 224, 3, 1)) |>
        x -> cflat(neural_model(x))
        return out
    end
end

```

Now, let's get the data. I downloaded the complete [cats and dogs](https://www.kaggle.com/c/dogs-vs-cats/data) dataset from Kaggle, and saved it locally in a directory called `train`. For purposes of this tutorial, I'm going to train and cross-validate on the training data because there is plenty of it. 

Let's sample 1000 images of cats and the same number of dogs for running these examples. I'm setting a random number generator seed here so I can reproduce my results. 

# Loading image paths 
```julia
myrng = MersenneTwister(234)
# create dataset for training and cross-validation 
cats = "train/".*StatsBase.sample(myrng, readdir("train/")[contains.(readdir("train/"), r"cat")], 1000, replace=false)
dogs = "train/".*StatsBase.sample(myrng, readdir("train/")[contains.(readdir("train/"), r"dog")], 1000, replace=false)
train_paths = [cats; dogs]
```

Now let's load our neural network model into memory and set up some simple functions. We will use the VGG19 neural network for this example. [VGG19](https://iq.opengenus.org/vgg19-architecture/) has 19 layers and was developed at Oxford University. There are many different neural network options that would work for our purposes here, but this one is considered a classic. Once it's loaded, it's easy to take a look at what the layers are by just indexing the layer from the function, e.g. `VGG19().layers[1]`.  We will pull the Dense layer of the network just before the output prediction. The prediction layer is indexing whatever ImageNet object the model thinks it sees, and for our purposes that is meaningless so we don't want to extract that layer! 

We then also create a prediction function based on the combination of our bottleneck pipeline and the pretrained model up to the Dense layer. This will pass our neural model into the bottleneck pipeline so that it can take an image and return a vector of embeddings. In order to make this all pretty clean, let's then pass the `capture_bottleneck` function into another function `capture_dogs_cats` that will allocate an arry to collect the results of our image processing and give us nice feedback on the progress. 

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
```

Now, we're ready to pass the images through the lower levels of the neural network. As I said earlier, we're going to be REAL stingy with data - only 1000 cats and dogs. Let's see how the model does against this small amount of data. 

We capture the dog and cat features, then we grab the labels from the file names in the dog/cat paths. 

```julia
# create a dataset for training 
dog_cat_features = capture_dogs_cats(vcat(dogs[1:1000], cats[1:1000]));

# create ternary function that can be broadcast 
label_dog_cat(path) = contains(path, r"cat") ? "cat" : "dog";
y = label_dog_cat.([dogs[1:100]; cats[1:100]]);

```

Now, it's time to train our Support Vector Machine (SVM). The SVM will leverage these embeddings to actually classify images. If you don't know what an SVM is, [the Wikipedia entry](https://en.wikipedia.org/wiki/Support-vector_machine) for SVM's is actually quite good. The essence of the SVM is that it determines a 'support vector' that maximizes the 'distance' between the cat and dog classes we're trying to classify. The support vector is the subset of the embeddings which display the maximum difference between the 'cat' and 'dog' examples in the data. 

We will train the model and use K-fold cross-validation to pump out the results. K-fold cross-validation essentially cuts our training data into equally-sized slices and reserves one of them for evaluating the model's performance. Then we'll spit out the model performance on those reserved slices. 

And the results are not bad! Keep in mind the cat/dog classes are perfectly balanced 50/50, and we used an absurdly small amount of data. 

```julia 
# SVM definition 
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

RSK = RepeatedStratifiedKFold(n_splits=6, n_repeats=1, random_state=3403)
out = cross_val_score(svm, dog_cat_features, y, cv = RSK.split(dog_cat_features,  y))

```

There are a number of things to learn from here. The next thing to try is retraining the upper layers. That would yield higher accuracy, but we'll save that for another post. 
