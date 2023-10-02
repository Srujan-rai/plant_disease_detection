import json
import base64
import io
import tensorflow as tf
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class DiseasePrediction(Resource):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = tf.keras.models.load_model("68 accuracy.h5")
            self.data_tuple = (
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
            self.ref = {index: disease for index, disease in enumerate(self.data_tuple[1])}
            with open('remedies.json', 'r') as remedies_file:
                self.remedies_data = json.load(remedies_file)

    def predict_disease_and_remedy(self, base64_image):
        try:
            # Decode the base64 image
            image_data = base64.b64decode(base64_image)
            image = tf.image.decode_image(image_data)
            image = tf.image.resize(image, (256, 256))
            img_array = tf.keras.applications.vgg19.preprocess_input(image)
            img_array = tf.expand_dims(img_array, axis=0)

            with self.graph.as_default():
                prediction_index = tf.argmax(self.model.predict(img_array))
                predicted_disease = self.ref[prediction_index.numpy()]

            # Get the remedy from the remedies_data JSON
            remedy_info = self.remedies_data.get(predicted_disease, 'No remedy information available')

            return {'result': predicted_disease, 'remedy': remedy_info}

        except Exception as e:
            return {'error': str(e)}

    def post(self):
        if 'base64_image' not in request.json:
            return {'error': 'No base64 image data provided'}

        base64_image = request.json['base64_image']

        result = self.predict_disease_and_remedy(base64_image)

        return result

api.add_resource(DiseasePrediction, '/predict')

if __name__ == '__main__':
    import os
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8000))  # Use the PORT environment variable provided by Render
    app.run(debug=False, host=host, port=port)
