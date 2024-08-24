from flask import Flask, render_template, request, redirect, url_for
import pickle
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Ensure the uploaded images are saved to a directory
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the file is present in the request
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file is valid
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = Image.open(file_path)
            img = img.resize((256, 256))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

            return render_template('result.html', filename=filename, result=result)

    return render_template('index.html')


# Function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to display uploaded image
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
