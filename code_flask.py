from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.preprocessing import image                  
from tqdm import tqdm
from extract_bottleneck_features import *
from glob import glob
import numpy as np
from flask import Flask
from flask import jsonify

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

ResNet50_model_dog = Sequential()
ResNet50_model_dog.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
ResNet50_model_dog.add(Dense(133, activation='softmax', input_shape=(6680, 1, 1, 2048)))

ResNet50_model_dog.load_weights('saved_models/weights.best.ResNet50.hdf5')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model_dog.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

app = Flask(__name__)

@app.route('/predict/<folder>/<filename>')
def predict(folder, filename):
    dog_breed = ResNet50_predict_breed('dogImages/train/' + folder + '/' + filename)

    return jsonify(
        dog_breed=dog_breed,
        filename=filename
    )