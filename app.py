from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the StandardScaler and machine learning model for water quality prediction
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model_water_quality = pickle.load(model_file)

@app.route('/')
def index():
    return "<center><h1>Flask App Water quality </h1></center>"

@app.route('/predict_water_quality', methods=['POST'])
def predict_water_quality():
    try:
        # Get input parameters from the request
        input_data = request.get_json(force=True)

        # Extract features and convert to float
        features = [float(input_data[param]) for param in ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']]

        # Apply StandardScaler
        scaled_features = scaler.transform([features])

        # Make predictions
        prediction = model_water_quality.predict(scaled_features)[0]

        # Convert prediction to a regular Python integer
        prediction = int(prediction)

        # Return the prediction as JSON
        return jsonify({'prediction_water_quality': prediction})

    except Exception as e:
        return jsonify({'error_water_quality': str(e)})

if __name__=="__main__":
    app.run(port=8000)
