from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
#from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys
import os
from scipy.misc import imread
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model = InceptionResNetV2(weights='imagenet')
model = ResNet50(weights='imagenet')

if sys.argv[1] == '--path':
    path = sys.argv[2]
    img = imread(path)
    x = img.astype(np.float32)

else:
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
