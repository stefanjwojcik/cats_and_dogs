using MLDatasets, Flux
using Plots, Images
using Statistics

# Create tensorboard logger
# TensorBoardLogger.TBLogger

using TensorBoardLogger, Logging

logger = TBLogger("content/log", tb_overwrite)

# Function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

# Callback to log information after every epoch
function TBCallback()
    param_dict = Dict{String, Any}()
    fill_param_dict!(param_dict, model, "")
    with_logger(logger) do
      @info "model" params=param_dict log_step_increment=0
      @info "train" loss=loss(xtrain, ytrain) acc=accuracy(xtrain, ytrain) log_step_increment=0
      @info "test" loss=loss(xtest, ytest) acc=accuracy(xtest, ytest)
    end
end
#--------------------

  # load full training set
train_x, train_y = MNIST.traindata(Float32)

# load full test set
test_x,  test_y  = MNIST.testdata(Float32)

# Viewing output 
colorview(Gray, train_x[:, :, 1]')

# RESHAPING 
# Reshape Data in order to flatten each image into a linear array
xtrain = Flux.flatten(train_x)
xtest = Flux.flatten(test_x)

# One-hot-encode the labels
ytrain, ytest = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9)

# Get the dimensions of train_x
(m, n, z) = size(train_x)

# Chain together functions!
model = Chain(
    Dense(28 * 28, 32, relu),
    Dense(32, 10),
    softmax
)

# Define mean squared error loss function
loss(x, y) = crossentropy(model(x), y)

# Define the accuracy 
accuracy(x, y) = Statistics.mean(Flux.onecold(model(x) |> cpu) .== Flux.onecold(y |> cpu))

# ADAM would be the perferred optimizer for serious deep learning
learning_rate = 0.01
opt = ADAM(learning_rate)

# Format your data
dataset = [(xtrain, ytrain)]

# Collect weights and bias for your model
parameters = Flux.params(model)

println("Old Loss = $(loss(xtrain, ytrain))")

# Train the model over 100 epochs
for epoch in 1:1000
    Flux.train!(loss, parameters, dataset, opt, cb = Flux.throttle(TBCallback, 5))
end


println("New Loss = $(loss(xtrain, ytrain))")
