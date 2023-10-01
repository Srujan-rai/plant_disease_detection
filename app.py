from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import base64
import numpy as np

app = Flask(__name__)


model = load_model("best_model.h5")


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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'})

    file = request.files['file']

    # Check if the file has a valid file name
    if file.filename == '':
        return jsonify({'error': 'No image file selected'})


    try:
        image = Image.open(file)
        image = image.resize((256, 256))
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)


        prediction_index = np.argmax(model.predict(img_array))
        predicted_disease = ref[prediction_index]

        return jsonify({'result': predicted_disease})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
