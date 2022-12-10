# Fine-tuning a model using Julia's Metalhead PACKAGES
# TODO: 
  # 1. Load the model using updated metalhead code 
  # 2. Load the paths to the images 
  # 3. Extract final layer of ResNet and add a Dense layer to fine tune 

using Flux, Images
using Flux: onehotbatch, @epochs, onecold
using Flux.Data: DataLoader
using StatsBase: sample, shuffle
using CSV, DataFrames, CUDA
using Pipe, Images

# Setup: getting the dogs and cats data 
# Download data from: https://www.kaggle.com/competitions/dogs-vs-cats/data
# store it in folders named 'train' and 'test1'

# Get data paths 
imgs = readdir("train")
# filter those that are not images
imgs = filter(x -> occursin(".jpg", x), imgs)

## Create a function to NORMALIZE IMAGES once they are read 
function normalize(img, nsize)
  # normalize images 
  @pipe img |> 
  RGB.(_) |>
  Images.imresize(_, nsize...) |> 
  (channelview(_) .* 255 .- 128)./128 |> 
  Float32.(permutedims(_, (3, 2, 1))[:,:,:,:])
end

## Load the pretrained model from Metalhead that will be used 
# load the model 
resnet = ResNet().layers

# Finally, the model looks something like:

model = Chain(
  resnet[1:end-2],
  Dense(2048, 256),
  Dense(256, 2),        # we get 2048 features out, and we have 2 classes
)

# send model to GPU 
model = model |> gpu
dataset = [gpu.(load_batch(10)) for i in 1:10]

# Define loss functions 
opt = ADAM()
loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)
accuracy(x, y) = Statistics.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# Define trainable parameters - just the tip 
ps = Flux.params(model[2:end])

# Train for two epochs 
@epochs 2 Flux.train!(loss, ps, dataset, opt)

# SHOW THE RESULTS 
imgs, labels = gpu.(load_batch(10))
display(model(imgs))

labels

############

(m, n) = size(xtrain)

# Train the model over 100_000 epochs
for epoch in 1:1000
    # Randomly select a entry of training data 
    dataset = [gpu.(load_batch(10)) for x in 1:10]
    xt, yt = dataset[1]

    # Implement Stochastic Gradient Descent 
    Flux.train!(loss, ps, dataset, opt)

    # Print loss function values 
    if epoch % 10 == 0
        println("Epoch: $(epoch)")
        @show loss(xt, yt)
        @show accuracy(xt, yt)
        println()
    end
end