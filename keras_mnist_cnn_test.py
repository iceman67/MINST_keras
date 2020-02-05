# https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b

import cv2
import h5py
from keras.models import load_model
#from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras import backend as K

#K.set_image_dim_ordering('th')

if(K.common.image_dim_ordering() == 'th'):

    input_tensor = Input(shape=(3, 299, 299))


#Step 4
# define dimensions of our input images.
img_width, img_height = 28, 28  # Here this is 28 ,28 because the shape of image is 28,28,3. You can input any shape greater than 28.
                                # You can give shape 150, 150. It will just take longer time for model to run. 

model_file = 'output/mnist-cnn-best.hdf5'
test_image_file = 'test/img_34.jpg'  # 2
#test_image_file = 'test/img_5.jpg'   # 3
#test_image_file = 'test/num.jpg'   # 4
#test_image_file = 'test/num4.jpg'   # 4

#Step 9
#Test on the image form test folders
test_model = load_model(model_file)
print('model = ', test_model.input_shape) 


#img = load_img(test_image_file,False,target_size=(img_width,img_height))
img = load_img(test_image_file,grayscale=True,target_size=(img_width,img_height,1))
x = img_to_array(img)
x = x.reshape([-1,28, 28,1])
print ('input = ',x.shape)

#x = x.astype('float32')
#x /= 255

preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
print(preds, prob)
