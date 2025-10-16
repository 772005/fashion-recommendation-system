from flask import Flask, request, render_template, send_from_directory, abort
import os
import numpy as np
import pickle
try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.layers import GlobalMaxPooling2D
    from tensorflow.keras.models import Sequential
except Exception:
    # If tensorflow is not available in the editor environment, imports will fail here.
    image = None
    ResNet50 = None
    preprocess_input = None
    GlobalMaxPooling2D = None
    Sequential = None
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load pickles with graceful error handling
def load_pickles():
    try:
        features = pickle.load(open("image_features_embedding.pkl", "rb"))
        files = pickle.load(open("img_files.pkl", "rb"))
        return features, files
    except Exception as e:
        print("Warning: could not load pickles:", e)
        return None, None

features_list, img_files_list = load_pickles()

# Build model lazily only if tensorflow available
model = None
if ResNet50 is not None:
    try:
        model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = Sequential([model, GlobalMaxPooling2D()])
    except Exception as e:
        print('Warning: could not initialize ResNet50 model:', e)

UPLOAD_FOLDER = 'uploader'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/image/<path:filename>')
def serve_image(filename):
    # Try to serve from several known directories
    candidates = [os.path.join(app.config['UPLOAD_FOLDER'], filename), filename, os.path.join('fashion_small', 'images', filename), os.path.join('sample', os.path.basename(filename))]
    for p in candidates:
        if os.path.exists(p):
            return send_from_directory(os.path.dirname(os.path.abspath(p)) or '.', os.path.basename(p))
    abort(404)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if model is None or features_list is None:
                return "Server is not ready: model or embeddings missing. Ensure TensorFlow is installed and pickles are generated.", 500

            features = extract_img_features(file_path, model)
            indices = recommendd(features, features_list)
            recommended_images = [img_files_list[i] for i in indices[0][1:]]
            # convert to URLs under /image/
            from urllib.parse import quote
            recommended_urls = ['/image/' + quote(path.replace('\\\\','/')) for path in recommended_images]
            return render_template('results.html', uploaded_image='/image/' + filename, recommended_images=recommended_urls)
    return render_template('index.html') 

def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normlized = flatten_result / norm(flatten_result)
    return result_normlized

def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

if __name__ == '__main__':
    app.run(debug=True)