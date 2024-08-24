from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])
        family_size = request.form['family_size']

        # Convert to model input format
        pclass_2 = 1 if pclass == 2 else 0
        pclass_3 = 1 if pclass == 3 else 0
        sex_male = sex
        embarked_Q = 1 if embarked == 1 else 0
        embarked_S = 1 if embarked == 2 else 0
        family_size_large = 1 if family_size == 'Large' else 0
        family_size_medium = 1 if family_size == 'Medium' else 0

        # Create a numpy array of the inputs
        features = np.array(
            [[age, fare, pclass_2, pclass_3, sex_male, embarked_Q, embarked_S, family_size_large, family_size_medium]])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Output the result
        result = 'Survived' if prediction == 1 else 'Not Survived'
        return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
