from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import json
import requests

# Initialize Flask app
app = Flask(__name__)

# Load and process data
data = pd.read_csv('modData_updated.csv')  # Update the path if needed
items = data["Item"].unique()

yield_data = []
vals = []

for item in items:
    temp_data = []
    temp_vals = []
    for i in range(len(data["Item"])):
        if item == data["Item"][i]:
            temp_data.append(data['Value'][i])
            temp_vals.append([data['median_temp'][i], data['med_precip'][i], data['med_soil_tmp'][i], data['med_soil_moist'][i]])
    yield_data.append(temp_data)
    vals.append(temp_vals)

def getEstimatedVals(i, temp, precip, soiltmp, soilmoist):
    regressor = Lasso(alpha=1e-30)
    xpoints = np.array([vals[i][0], vals[i][1], vals[i][2]])
    ypoints = np.array(yield_data[i])
    ypoints = ypoints[:3]
    regressor.fit(xpoints, ypoints)
    tempewr = np.array([temp, precip, soiltmp, soilmoist]).reshape(1, -1)
    return np.average(regressor.predict(tempewr))

def returnBestValue(temp, precip, soil_tmp, soil_moist):
    temp_list = []

    for i in range(len(items)):
        try:
            predictedVal = getEstimatedVals(i, temp, precip, soil_tmp, soil_moist)
            temp_list.append((predictedVal, i))
        except:
            temp_list.append((0, i))

    temp_list.sort(reverse=True, key=lambda x: x[0])
    top_5 = temp_list[:10]

    result = []
    for val, index in top_5:
        result.append({
            "id": index,  # Include the ID (index)
            "crop": items[index],
            "predicted_yield": val
        })

    return json.dumps(result, indent=2)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temp = data.get('temp')
    precip = data.get('precip')
    soil_tmp = data.get('soil_tmp')
    soil_moist = data.get('soil_moist')

    if temp is None or precip is None or soil_tmp is None or soil_moist is None:
        return jsonify({"error": "Missing parameters"}), 400

    result = returnBestValue(temp, precip, soil_tmp, soil_moist)
    return jsonify(json.loads(result))

@app.route('/future_weather', methods=['GET'])
def get_future_weather_data():
    lat = request.args.get('latitude', type=float)
    lon = request.args.get('longitude', type=float)

    print(f"Received latitude: {lat}, longitude: {lon}")  # Debugging line

    if lat is None or lon is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    # Build the URL for the Open Meteo API for future weather data
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,soil_temperature_0cm,soil_moisture_0_to_1cm&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Extract hourly data
        temperatures = data['hourly']["temperature_2m"]
        precipitations = data['hourly']["precipitation"]
        soil_temperatures = data['hourly']["soil_temperature_0cm"]
        soil_moistures = data['hourly']["soil_moisture_0_to_1cm"]

        # Calculate averages
        weather_data = {
            "average_temperature": np.mean(temperatures),  # Use numpy to calculate average
            "average_precipitation": np.mean(precipitations),
            "average_soil_temperature": np.mean(soil_temperatures),
            "average_soil_moisture": np.mean(soil_moistures)
        }

        return jsonify(weather_data)

    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")  # More detailed error log
        return jsonify({"error": str(e)}), 500


