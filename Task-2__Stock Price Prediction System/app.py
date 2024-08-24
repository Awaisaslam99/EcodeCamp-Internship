from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Retrieve input values from form and ensure they are floats
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            volume = float(request.form['volume'])

            # Prepare data for prediction
            input_features = np.array([[open_price, high_price, low_price, volume]])

            # Make prediction
            prediction = model.predict(input_features)[0]

            # Render result template with prediction
            return render_template('result.html', prediction=prediction)
        except Exception as e:
            # Handle potential errors and print the error message
            print(f"Error: {e}")
            return redirect(url_for('index'))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
