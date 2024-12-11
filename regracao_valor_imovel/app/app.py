from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """
    Serve the HTML frontend.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict median house value based on user input.
    """
    try:
        # Extract data from request
        data = request.json
        # Convert data to numpy array (ensure correct feature order)
        features = np.array([[
            data['longitude'],
            data['latitude'],
            data['housing_median_age'],
            data['total_rooms'],
            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income'],
            data['ocean_proximity'],  # Add ocean_proximity here
            data['rooms_per_household'],
            data['bedrooms_per_room']
        ]])
        # Make prediction
        prediction = model.predict(features)[0]
        return jsonify({'predicted_median_house_value': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
