######################
# PACKAGES
#########################
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON
using CUDA


######################
# default parameters
Base.@kwdef mutable struct Args
  η = 3e-4             ## learning rate
  λ = 0                ## L2 regularizer param, implemented as weight decay
  batchsize = 128      ## batch size
  epochs = 10          ## number of epochs
  seed = 0             ## set seed > 0 for reproducibility
  use_cuda = true      ## if true use cuda (if available)
  infotime = 1 	     ## report every `infotime` epochs
  checktime = 5        ## Save the model every `checktime` epochs. Set to 0 for no checkpoints.
  tblogger = true      ## log training with tensorboard
  savepath = "runs/"   ## results path
end

######################
function get_data(args)
  xtrain, ytrain = MLDatasets.MNIST(:train)[:]
  xtest, ytest = MLDatasets.MNIST(:test)[:]

  xtrain = reshape(xtrain, 28, 28, 1, :)
  xtest = reshape(xtest, 28, 28, 1, :)

  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
  test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
  
  return train_loader, test_loader
end


function LeNet5(; imgsize=(28,28,1), nclasses=10) 
  out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
  
  return Chain(
          Conv((5, 5), imgsize[end]=>6, relu),
          MaxPool((2, 2)),
          Conv((5, 5), 6=>16, relu),
          MaxPool((2, 2)),
          flatten,
          Dense(prod(out_conv_size), 120, relu), 
          Dense(120, 84, relu), 
          Dense(84, nclasses)
        )
end