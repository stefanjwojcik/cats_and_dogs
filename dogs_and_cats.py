import os
import re
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import random

# load the model 
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)

# generate the paths:
def get_regex_list(mylist, myregex):
    r = re.compile(myregex)
    newlist = list(filter(r.match, mylist)) # Read Note below
    return(newlist)

def get_features_from_image(mymod, image_path=None):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = mymod.predict(x)
    features_raw = np.squeeze(predictions)
    return(features_raw)


# Set up training data 
os.chdir("train")
all_paths = os.listdir()
dogs = random.sample(get_regex_list(all_paths, "dog"), 2500)
cats = random.sample(get_regex_list(all_paths, "cat"), 2500)
train_paths = np.concatenate((dogs, cats))

nb_features = 2048
i = 0
# Create data receptacles for plain jane and face-segmented images
features_plain = np.empty((len(dogs)+len(cats), nb_features))
# Create a for-loop to create training features for plain jaine and face-segmented
for image_path in train_paths:
    features_plain[i,:] = get_features_from_image(nn_model, image_path)
    i += 1

### train
r = re.compile("dog|cat")
pet_labels = [r.search(x).group(0) for x in train_paths]

svm = LinearSVC(C=.01, loss='squared_hinge', penalty='l2', multi_class='ovr', random_state = 35552)
clf = CalibratedClassifierCV(svm) 
clf.fit(features_plain, np.array(pet_labels))

RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=36851234)
out = cross_val_score(clf, features_plain, np.array(pet_labels), cv = RSK.split(features_plain, np.array(pet_labels)))

# Bits to save for JULIA

# features_processed
np.savetxt("../python_processed_files/cat_dog_features.csv", features_plain, delimiter=",")
# labels 
np.savetxt("../python_processed_files/cat_dog_labels.csv", pet_labels, delimiter=",", fmt='%s')
# training paths
np.savetxt("../python_processed_files/cat_dog_train_paths.csv", train_paths, delimiter=",", fmt='%s')

######################################################
# Create class activation map for ResNet50

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

# Here is the model: 
resmod = ResNet50(weights='imagenet')

myimg = image.load_img("train/cat.1.jpg", target_size=(224, 224))

# pring the image 
plt.imshow(myimg)

def get_class_activation_map(model, img):
    """ 
    this function computes the class activation map
    
    Inputs:
        1) model (tensorflow model) : trained model
        2) img (numpy array of shape (224, 224, 3)) : input image
    """
    # expand dimension to fit the image to a network accepted input size
    img = np.expand_dims(img, axis=0)
    # predict to get the winning class
    predictions = model.predict(img)
    label_index = np.argmax(predictions)

    # Get the 2048 input weights to the softmax of the winning class.
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]
    
    # get the final conv layer
    final_conv_layer = model.get_layer("conv5_block3_out")
    
    # create a function to fetch the final conv layer output maps (should be shape (1, 7, 7, 2048)) 
    get_output = K.function([model.layers[0].input],[final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    
    # squeeze conv map to shape image to size (7, 7, 2048)
    conv_outputs = np.squeeze(conv_outputs)
    
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (32, 32, 1), order=1) # dim: 224 x 224 x 2048

    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), class_weights_winner).reshape(224,224) # dim: 224 x 224
    
    # return class activation map
    return final_output, label_index

