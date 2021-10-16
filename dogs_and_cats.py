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

# train_paths

