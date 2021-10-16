## UTILS FOR IMAGE PROCESSING with RESNET50 and FLUX 

# PYTHON MODULES REQUIRED  
py"""
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
"""

# This is a PYTHON-BASED PIPELINE FOR COMPARISON 
py"""
#def get_features_from_image(mymod, image_path=None):
#    img = image.load_img(image_path, target_size=(224, 224))
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#    predictions = mymod.predict(x)
#    features_raw = np.squeeze(predictions)
#    return(x)

_ = VGG19(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('fc2').output)
"""

# Make the python functions available in base Julia 
#function pyproc(x)
#    # python 
#    xc = deepcopy(x)
#    xc = py"get_features_from_image"(py"nn_model", xc)
#    return xc
#end

# Make this python process available in Julia 
function py_process_input(image_array)
    image_array_cx = deepcopy(image_array)
    image_array_cx .= py"preprocess_input"(image_array_cx)
    return image_array_cx
end

### JULIA FUNCTIONS (first draws on imagenet means )

imagenet_means = mean(py_process_input(zeros(224, 224, 3)), dims=(1, 2))

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

# utility function to flatten arrays to compare them 
function cflat(x::AbstractArray)
    collect(Iterators.flatten(x))
end