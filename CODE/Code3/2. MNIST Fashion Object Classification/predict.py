# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:16:52 2021

@author: noopa
"""

# make a prediction for a new image.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
def load_image(filename):
    #load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    #convert to array
    img = img_to_array(img)
    #reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    #prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

img1 = mpimg.imread('sample_image.png')
imgplot = plt.imshow(img1)
plt.show()
img = load_image("sample_image.png")
model = load_model('final_model1.h5')
# predict the class
result = model.predict_classes(img)
if result[0] == 0:
    print("Top")
elif result[0] == 1:
    print("Trouser")
elif result[0] == 2:
    print("Pullover")
elif result[0] == 3:
    print("Dress")
elif result[0] == 4:
    print("Coat")
elif result[0] == 5:
    print("Sandal")
elif result[0] == 6:
    print("Shirt")
elif result[0] == 7:
    print("Sneaker")
elif result[0] == 8:
    print("Bag")
elif result[0] == 9:
    print("Ankle Boot")
else:
    print("Not in the list")