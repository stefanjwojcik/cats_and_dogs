## UTILS FOR IMAGE PROCESSING with RESNET50 and FLUX 

using Metalhead
using Flux
using Images
#using OffsetArrays
#using ImageMagick
# MUST: add CUDA@1.3.3
using CUDA
using Pipe
using ProgressMeter
#using PyCall

# utility function to flatten arrays to compare them 
function cflat(x::AbstractArray)
    collect(Iterators.flatten(x))
end

# The actual function that will be used 
"""
This is a function to scale images to the ImageNet means. The purpose is to scale images to match ImageNet. 
Python also switches the 1st and 3rd RBG channels sometimes, so this function does that too. 
"""
function jimage_net_scale!(dx::AbstractArray, channels_last=false)
    imagenet_means = [-103.93899999996464, -116.77900000007705, -123.67999999995286]
    #dx = copy(x)
    # swap R and G channels like python does - only during channels_last 
    if channels_last
        dx[:, :, :, 1], dx[:, :, :, 3] = dx[:, :, :, 3], dx[:, :, :, 1]
    end
    dx[:, :, :, 1] .+= imagenet_means[1]
    dx[:, :, :, 2] .+= imagenet_means[2]
    dx[:, :, :, 3] .+= imagenet_means[3]
    #return cor(collect(Iterators.flatten(dx)),collect(Iterators.flatten(py_scaled_image)))
    return(dx)
end

"""
This is the function that takes a function and returns a function for processing each image.  
"""
function create_bottleneck_pipeline(neural_model)
    function capture_bottleneck(image_path)
        out = @pipe load(image_path) |> #
        x -> imresize(x, 224, 224) |> # resize the image to imagenet dims
        x -> Float32.(channelview(x) * 255) |> # drop to an array
        x -> permutedims(x, [2, 3, 1]) |> # swap ordering of dimensions 
        x -> reshape(x, (1, 224, 224, 3) ) |> # Python style for comparison sake 
        x -> jimage_net_scale!(x) |>
        #x -> reshape(x, (224, 224, 3, 1)) |>
        x -> cflat(neural_model(x))
        return out
    end
end

#nn_model = VGG19().layers[1:25];
#capture_bottleneck = create_bottleneck_pipeline(nn_model);


# TODO: Fix this function, cut out the face segmentation for separate step 
function generate_resnet_features(image_paths)
    # Progress Meter just to see where we are in the process 
    p = ProgressMeter.Progress(length(image_paths)) # total of iterations, by 1
    ProgressMeter.next!(p)
    failed_cases = String[]

    # Creating empty arrays to store the results 
    features_out = CUDA.zeros(Float32, 2048, length(image_paths)) # create base 

    for (i, img_key) in enumerate(keys(image_paths))
        #printstyled(img_key*" \n", color=:green)
        try
            raw_img = load(img_key)
            img_features = capture_bottleneck(raw_img)
            @inbounds features_out[:, i] .= img_features[:, 1]
        catch
            #@warn "$img_key failed"
            push!(failed_cases, img_key)
            @inbounds features_out[:, i] .= CUDA.zeros(2048, )
        end

        ProgressMeter.next!(p)
    end
    # Write out the files
    out = (features_out, failed_cases)
    printstyled("DONE \n", color=:blue)
    return out
end
