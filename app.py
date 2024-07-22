from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load precomputed feature vectors and filenames
feature_list = np.array(pickle.load(open('featurevector.pkl', "rb")))
filename = pickle.load(open('filenames.pkl', "rb"))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Define the NearestNeighbors model
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                # Extract features and find similar images
                features = extract_feature(file_path, model)
                distances, indices = neighbors.kneighbors([features])

                similar_images = [filename[index] for index in indices[0][1:5]]
                return render_template('result.html', image_path=file_path, similar_images=similar_images)
            except Exception as e:
                print(f"Error: {e}")
                return redirect(request.url)

    return render_template('index.html')
