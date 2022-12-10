
# Observations: 
    # - final model in python vs. methalhead julia doesn't make a huge diff in performance
    # - scaling to imagenet means doesn't seem to make a huge differences 

    py"""
    import numpy as np
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg19 import preprocess_input
    _ = ResNet50(weights='imagenet')
    nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)
    
    def get_features_from_image(mymod, image_path=None):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = mymod.predict(x)
        features_raw = np.squeeze(predictions)
        return(features_raw)
    
    def test_preprocess(image_path=None):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return(x)    
    """
    
    # Download images from Kaggle and save in a training file: 
        # download() from https://www.kaggle.com/c/dogs-vs-cats/data?select=train.zip
    
    # create a list of cat and dog images to sample 
    cats = "train/".*rand(readdir("train/")[contains.(readdir("train/"), r"cat")], 1000)
    dogs = "train/".*rand(readdir("train/")[contains.(readdir("train/"), r"dog")], 1000)
    
    # image_net_scale moved to cats_and_dogs/utils.jl
    
    # load each image and pass it through the VGG net, save the output
    function capture_bottleneck(array_of_image_paths, nmodel)
        p = ProgressMeter.Progress(length(array_of_image_paths)) # total of iterations, by 1
        ProgressMeter.next!(p)
        # create bottleneck features object and run 
        #bottleneck_features = CUDA.zeros(Float32, 2048, length(array_of_image_paths))
        bottleneck_features = zeros(Float32, 2048, length(array_of_image_paths))
        for (i, pic) in enumerate(array_of_image_paths)
            #j_image = Array{Float32, 4}(undef, 224, 224, 3, 1) 
            j_image_load = @pipe imresize(load(pic), 224, 224) |> 
                            x -> Float32.(channelview(x) ) |> 
                            x -> permutedims(x, [3,2,1]) |>
                            x -> reshape(x, 1, 224, 224, 3) |> 
                            x -> image_net_scale(x)
            #j_image_load[:, :, 1:3, 1] .=  # resize the image 
            proc_pic = @pipe j_image_load |> 
                #x -> py"nn_model.predict"(x)
                x -> nmodel(x)
                #x -> (model(x) |> Flux.gpu)[1,1,:,1]
            @inbounds bottleneck_features[:, i] .= proc_pic[1, :]
            ProgressMeter.next!(p)
        end
        return bottleneck_features
    end
    
    function get_features_from_image(img, nmodel)
        j_image_load = @pipe imresize(load(img), 224, 224) |> 
                        x -> Float32.(channelview(x) ) |> 
                        x -> permutedims(x, [2,3,1]) |>
                        x -> reshape(x, 224, 224, 3, 1) |> 
                        x -> image_net_scale(x)
        proc_pic = @pipe j_image_load |> 
            x -> nmodel(x)
        return proc_pic
    end
    
    get_features_from_image("train/"*pet_paths[1,1], mymod)
    
    ## Process cat and dog images 
    #mod = ResNet50(pretrain=true)
    #mymod = mod.layers[1:2][1]
    
    dog_features = capture_bottleneck_python(dogs, py"nn_model.predict")
    cat_features = capture_bottleneck_python(cats, py"nn_model.predict")
    
    # create final dataset for training 
    all_features = [dog_features'; cat_features']
    y = contains.([dogs; cats], r"cat")
    
    # Run SVM with bottleneck as features 
    
    ## LOADING SCIKITLEARN 
    using ScikitLearn
    @sk_import svm: LinearSVC
    @sk_import model_selection: RepeatedStratifiedKFold 
    @sk_import model_selection: cross_val_score
    svm = LinearSVC(C=.01, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
    svm.fit(Float32.(all_features), convert(Array, y))
    
    # accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
    RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=36851234)
    out = cross_val_score(svm, Float32.(all_features), convert(Array, y), cv = RSK.split(Float32.(all_features), convert(Array, y)))
    
    
    ############ Using python processed features to test the scikit-learn
    features_plain = CSV.read("python_processed_files/cat_dog_features.csv", DataFrame, header=false)
    pet_labels = CSV.read("python_processed_files/cat_dog_labels.csv", DataFrame, header=false)
    pet_paths = CSV.read("python_processed_files/cat_dog_train_paths.csv", DataFrame, header=false)
    
    
    svm = LinearSVC(C=.01, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
    fit!(svm, Matrix(features_plain), pet_labels[!, 1])
    
    # accuracy is similar to python here, so the error must be in processing somewhere
    RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=36851234)
    out = cross_val_score(svm, Matrix(features_plain), pet_labels[!, 1], cv = RSK.split(Matrix(features_plain), pet_labels[!, 1]))
    
    #********
    ############# CHECKING PROCESSING through the PYCALL API
    
    # - Check for features -> NN
        # all python through julia compared to all python (very small diffs)
    sum(py"get_features_from_image"(py"nn_model",  "train/"*pet_paths[1, 1])) # equals 787.91064f0
        # sum(get_features_from_image(nn_model, train_paths[0])) #equals 787.9107387939221
    
    mymod = ResNet50(pretrain=true)
    mymod = mymod.layers[1:2][1]
    get_features_from_image("train/"*pet_paths[1, 1], mymod)
    
    
    # - Testing preprocessing system 
    allones = ones(1, 224, 224, 3)
    py_out = py"preprocess_input"(allones)
    ju_out = image_net_scale(reshape(allones, 224, 224, 3, 1))
    # test
    this[1, :, 1, 1] == hi[:, 1, 1, 1]
    this[1, 1, 1, :] == hi[1, 1, :, 1]
    
    ### OLD CATS AND DOGS CODE 
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