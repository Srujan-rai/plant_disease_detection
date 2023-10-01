from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

import numpy as np
model=load_model("best_model.h5")

data_tuple = (
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [
        'Cherry___Powdery_mildew',
        'Cherry___healthy',
        'Pepper___Bacterial_spot',
        'Pepper___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
)

ref = {index: disease for index, disease in enumerate(data_tuple[1])}
def prediction(path):
  img=load_img(path,target_size=(256,256))
  i=img_to_array(img)
  im=preprocess_input(i)
  img=np.expand_dims(im,axis=0)
  pred=np.argmax(model.predict(img))
  print(f"the image is {ref[pred]}")

path="00b7df55-c789-43d6-a02e-a579ac9d07e6___FREC_Pwd.M 4748.JPG"
prediction(path)



