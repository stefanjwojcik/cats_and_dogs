using Images
using ImageTransformations
using CUDA
using PyCall
using Pipe
using ProgressMeter
using Statistics 
#using ImageMagick

# Set home directory 
cd("/home/swojcik/github/cats_and_dogs")

# create a list of cat and dog images to sample 
cats = "train/".*rand(readdir("train/")[contains.(readdir("train/"), r"cat")], 1000)
dogs = "train/".*rand(readdir("train/")[contains.(readdir("train/"), r"dog")], 1000)

## Python Vs. Julia running tests on image processing 
# DEFINE VGG model and Python equivalent (see utils.jl)
juresmod = VGG19().layers[1:25];

## ALL PYTHON INTERMEDIARY OUTPUT 
pyload = py"image.load_img"(dogs[1], target_size=(224, 224));
pyarray = py"image.img_to_array"(pyload);
pyarray_exp = py"np.expand_dims"(pyarray, axis=0);
pyarray_proc = py"preprocess_input"(pyarray_exp);
py_predictions = py"nn_model".predict(pyarray_proc);
py_features = py"np.squeeze"(py_predictions);

# ALL JULIA INTERMEDIARY OUTPUT 
juload = @pipe load(dogs[1]) |> imresize(_, 224, 224);
juarray = @pipe channelview(copy(juload) * 255) |> permutedims(_, [2,3,1]);
juarray_exp = reshape(copy(juarray), (1, 224,224,3));
juarray_proc = image_net_scale(juarray_exp);
ju_predictions = @pipe reshape(juarray_proc, (224, 224, 3, 1) ) |> 
    juresmod(_);

# possible julia models 
#ju_prepped = reshape(juarray_proc, (224, 224, 3, 1) );
#jupred25 = VGG19().layers[1:25](ju_prepped);
#jupred24 = VGG19().layers[1:24](ju_prepped);
#jupred23 = VGG19().layers[1:23](ju_prepped);

#cor(cflat(py_predictions), cflat(jupred25))
#cor(cflat(py_predictions), cflat(jupred24))
#cor(cflat(py_predictions), cflat(jupred23))

function get_features_from_image(mymod, image_path=None)
    out = @pipe load(image_path) |> 
    x -> imresize(x, 224, 224) |>
    x -> channelview(x) * 255 |> 
    x -> permutedims(x, [2, 3, 1]) |>
    x -> image_net_scale(x) |>
    x -> reshape(x, (224, 224, 3, 1) ) #|> 
    #x -> cflat(mymod(x))
    return out
    #predictions = mymod(x)
    #features_raw = np.squeeze(predictions)
    #return(features_raw)
end

function py_nn_model(x)
    # python 
    xc = deepcopy(x)
    xc = py"np.expand_dims"(xc, axis=0)
    return py"nn_model.predict"(xc)
end

#Try processing random images and comparing their output with VGG19
# the format for an image is (224, 224, 3, 1)
for i in 1:20
    this = rand(Float32, 224, 224, 3, 1) * 255
    ju_scaled_img = juresmod(this);
    py_scaled_img = py_nn_model(this);
    println(cor(cflat(ju_scaled_img),cflat(py_scaled_img)))
end

### SCALING 

# Predefine some python functions here to avoid object mutation
function py_process_input(image_array)
    image_array_cx = deepcopy(image_array)
    image_array_cx .= py"preprocess_input"(image_array_cx)
    return image_array_cx
end

# test it 
myrand = rand(Float32, 224, 224, 3) * 255;
size(py_process_input(myrand));
mean(py_process_input(myrand), dims=(1,2))

# Candidate Julia function for scaling : 

# Function to compare hand-rolled preprocessing and python's (issue previously was this function mutating in place )
imagenet_means = mean(py_process_input(zeros(224, 224, 3)), dims=(1, 2))

function pyscale_set(imagenet_means)
    #imagenet_means = [103.93899999996464, 116.77900000007705, 123.67999999995286]
    function pyscale(x::AbstractArray)
        dx = copy(x)
        # swap R and G channels like python does - only during channels_last 
        dx[:, :, 1], dx[:, :, 3] = dx[:, :, 3], dx[:, :, 1]
        dx[:, :, 1] .+= imagenet_means[1]
        dx[:, :, 2] .+= imagenet_means[2]
        dx[:, :, 3] .+= imagenet_means[3]
        #return cor(collect(Iterators.flatten(dx)),collect(Iterators.flatten(py_scaled_image)))
        return(dx)
    end
end

# create scaling function based on imagenet means 
pyscale = pyscale_set(imagenet_means)

# flatten arrays to compare them 
function cflat(x::AbstractArray)
    collect(Iterators.flatten(x))
end

# random - previously failed apparently due to mutating in place (trying to fix with deepcopy)
    # this chunk below shows correlation at .999
myrand = rand(224, 224,3) *255;
myjurandscale = pyscale(myrand);
mypyrandscale = py_process_input(myrand);
cor(cflat(myjurandscale),cflat(mypyrandscale))
    
### Determined that python swaps channels RBG -> GBR
cor(cflat(myjurandscale[:, :, 1]),cflat(mypyrandscale[:, :, 3]))
#cor(cflat(myjurandscale[2, :, :]),cflat(mypyrandscale[2, :, :]))
#cor(cflat(myjurandscale[3, :, :]),cflat(mypyrandscale[1, :, :]))


## The models to be tested 
    # julia 
jumod = ResNet50(pretrain=true)
jumod = mymod.layers[1:2][1]

# Loading tests - note that Julia format is 'channels-first', so indicate this in python to match 
py_loaded = py"image.load_img"(testimg, target_size=(224, 224)) |>  
        x -> py"image.img_to_array"(x, data_format="channels_first");
ju_loaded = imresize(load(testimg), 224, 224) |> 
        x -> Float32.(channelview(x) * 255);
cor(collect(Iterators.flatten(ju_loaded)),collect(Iterators.flatten(py_loaded)))

# all ones - perfectly correlated 
myones = ones(3, 224, 224) * 255;
ju_scale = pyscale(myones);
py_scale = py_process_input(myones);
cor(collect(Iterators.flatten(ju_scale)),collect(Iterators.flatten(py_scale)))


# test image - some problems here - ??? consistently low correlation 
test_image_loaded = Float32.(channelview(imresize(load(dogs[1]), 224, 224)));
ju_scaled_img = pyscale(test_image_loaded);
py_scaled_img = py_process_input(test_image_loaded);
cor(collect(Iterators.flatten(ju_scaled_img)),collect(Iterators.flatten(py_scaled_img)))

#Try loading a bunch of different images and comparing their output 
for dog in dogs[1:20]
    test_image_loaded = @pipe load(dog) |> 
        x -> imresize(x, 224, 224) |>
        x -> channelview(x) * 255
    ju_scaled_img = pyscale(test_image_loaded);
    py_scaled_img = py_process_input(test_image_loaded);
    println(cor(collect(Iterators.flatten(ju_scaled_img)),collect(Iterators.flatten(py_scaled_img))))
end


# Model processing tests: now take the same random array

