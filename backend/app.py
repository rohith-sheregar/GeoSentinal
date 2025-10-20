# app.py
from flask import Flask, send_file, jsonify, request, make_response, send_from_directory
from flask_cors import CORS
import numpy as np
import io
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import random
import threading
import time
from datetime import datetime, timedelta

from risk import compute_risk, create_polygon_mask
from ml_model import RockfallPredictor
from sensor_monitor import SensorMonitor

app = Flask(__name__, static_folder='../frontend', static_url_path='', template_folder='../frontend')
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize components
predictor = RockfallPredictor()
from sensor_manager import SensorManager
sensor_manager = SensorManager()

def alert_handler(alert_data):
    print(f"Alert received: {alert_data['message']}")
    # Here you could integrate with SMS/email notification systems

sensor_manager.register_alert_callback(alert_handler)
sensor_manager.start_monitoring()

THIS_DIR = os.path.dirname(__file__)
DEM_PATH = os.path.join(THIS_DIR, "dem.npy")

if not os.path.exists(DEM_PATH):
    from dem_generator import generate_dem
    print("Generating DEM...") # Added print statement
    dem = generate_dem()
    np.save(DEM_PATH, dem)
    print("DEM generated and saved.") # Added print statement
else:
    print("Loading existing DEM...") # Added print statement
    dem = np.load(DEM_PATH)
    print("DEM loaded.") # Added print statement

BOUNDS = [12.95, 77.55, 13.05, 77.65]

@app.route("/")
def home():
    print("Home route accessed")
    return send_from_directory(app.template_folder, 'index.html')

@app.route("/api/sensor-data", methods=['GET'])
def get_sensor_data():
    return jsonify(sensor_manager.get_current_readings())

@app.route("/api/historical-data", methods=['GET'])
def get_historical_data():
    hours = request.args.get('hours', default=24, type=int)
    data = sensor_manager.get_historical_data(hours)
    return jsonify({
        'data': data,
        'timestamps': [
            (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
            for i in range(hours-1, -1, -1)
        ]
    })

@app.route("/api/predict", methods=['POST'])
def predict_rockfall():
    try:
        data = request.json
        current_sensors = sensor_manager.get_current_readings()
        
        # Combine DEM, image, and sensor data for prediction
        prediction = predictor.predict(
            data.get('dem_data'),
            data.get('image_data'),
            current_sensors
        )
        
        return jsonify({
            'probability': float(prediction[0]),
            'timestamp': datetime.now().isoformat(),
            'sensor_data': current_sensors
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'probability': 0.0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route("/risk_map.png")
def risk_map():
    # Get real sensor data
    sensor_data = sensor_manager.get_current_readings()
    rainfall = sensor_data.get('rainfall', 0)
    historical_data = sensor_manager.get_historical_data(24)
    duration = 24  # Use last 24 hours of data

    print(f"Generating risk map with real factors: Rainfall={rainfall}mm, Duration={duration}h")

    try:
        # --- Dynamic Heatmap Generation ---
        # The `compute_risk` function from risk.py is now used.
        # It expects `rainfall` as a parameter. We'll pass our simulated value.
        # We'll also pass a dummy NDVI for now as it's in the function signature.
        risk_data = compute_risk(dem, rainfall=rainfall, ndvi=None)

        # Normalize risk_data to 0-1 for colormap.
        # The `norm` function from risk.py can be used here if imported, or do it manually.
        if risk_data.max() > risk_data.min():
            normalized_risk = (risk_data - risk_data.min()) / (risk_data.max() - risk_data.min())
        else:
            normalized_risk = np.zeros_like(risk_data)

        # Use a colormap to convert normalized risk values to colors
        cmap = cm.jet # Other options: cm.hot, cm.viridis
        colored_image_array = cmap(normalized_risk)

        # Convert to PIL Image (RGBA 0-1 float -> 0-255 uint8)
        img = Image.fromarray((colored_image_array * 255).astype(np.uint8))

        # Save image to a byte buffer
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr.seek(0)

        # Close the plot to prevent memory leaks
        plt.close('all')

        print("Dynamic risk map generated successfully.")
        return send_file(byte_arr, mimetype='image/png')

    except Exception as e:
        print(f"Error generating risk map: {e}")
        return make_response(jsonify({"error": f"Failed to generate risk map: {str(e)}"}), 500)

@app.route("/metrics")
def metrics():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)

    # Simulate real-time factors that influence risk
    rainfall = random.randint(0, 500)
    duration = random.randint(1, 24)

    if lat is not None and lng is not None:
        # In a real app, you'd map lat/lng to an index in the DEM and get a specific risk.
        # For now, we simulate a more informed risk based on factors.
        base_risk = (rainfall / 500 + duration / 24) / 2
        max_risk = min(1.0, base_risk + random.uniform(0.1, 0.3))
        mean_risk = min(1.0, base_risk + random.uniform(0, 0.15))
        explanation = f"Risk for location ({lat:.4f}, {lng:.4f}). Factors: Rainfall={rainfall}mm, Duration={duration}h."
    else:
        max_risk = 0.5  # Default value
        mean_risk = 0.3  # Default value
        explanation = "Default risk evaluation. Provide lat/lng for specific analysis."

    return jsonify({
        "max_risk": round(max_risk, 2),
        "mean_risk": round(mean_risk, 2),
        "explanation": explanation,
        "factors": {
            "rainfall": rainfall,
            "duration": duration
        }
    })

@app.route("/bounds")
def bounds():
    s, w, n, e = BOUNDS
    return jsonify(south=s, west=w, north=n, east=e)

@app.route("/legend")
def legend_info():
    # Provides colors for the frontend legend, corresponding to the 'jet' colormap
    # High (1.0) -> Red, Mid (0.5) -> Yellow, Low (0.0) -> Blue/Green
    return jsonify({
        "high": "#FF0000",   # Red
        "medium": "#FFFF00", # Yellow
        "low": "#008000"     # Green (approximating part of the jet map)
    })

@app.route('/risk_graph', methods=['GET', 'POST'])
def risk_graph():
    # Default factors for the initial page load (GET request)
    rainfall = 100
    duration = 5
    geology_factor = 0.3

    # Use factors from the selected area if provided (POST request)
    if request.method == 'POST':
        data = request.get_json().get("factors", {})
        rainfall = data.get('Rainfall (mm)', rainfall)
        duration = data.get('Duration (hrs)', duration)
        geology_factor = data.get('Geology Instability', geology_factor)

    # Simulate a risk trend based on the factors.
    # Higher factors result in a higher starting risk and a more volatile trend.
    base_risk = (rainfall / 500.0 + duration / 72.0 + geology_factor) / 3.0
    time_series = [max(0, min(1.0, base_risk + random.uniform(-0.15, 0.15)))]
    for _ in range(9):
        previous_risk = time_series[-1]
        # Trend is influenced by the base risk, with some randomness
        trend = (base_risk - 0.5) * 0.1
        next_risk = previous_risk + trend + random.uniform(-0.05, 0.05)
        time_series.append(max(0, min(1.0, next_risk)))

    timestamps = [f"T-{9-i}h" for i in range(10)]

    return jsonify({
        "timestamps": timestamps,
        "risk_values": [round(r, 2) for r in time_series]
    })

@app.route('/polygon_heatmap', methods=['POST'])
def polygon_heatmap():
    data = request.get_json()
    # Leaflet sends coordinates nested, so we access the first element.
    coordinates = data.get('coordinates', [[]])[0]

    if not coordinates or len(coordinates) < 3:
        return make_response(jsonify({"error": "Insufficient coordinates for polygon."}), 400)

    try:
        # --- Dynamic Polygon Heatmap Generation ---
        # Generate a unique set of factors for this specific request
        rainfall = random.randint(0, 500)
        duration = random.randint(1, 72) # e.g., up to 3 days of continuous rain
        geology_factor = random.uniform(0.1, 1.0) # Simulate soil/rock type (1.0 is unstable)
        
        print(f"Generating polygon heatmap with factors: Rain={rainfall}, Duration={duration}, Geology={geology_factor:.2f}")
        
        # 1. Create the polygon mask using the helper function
        mask = create_polygon_mask(dem.shape, coordinates, BOUNDS)

        # 2. Compute risk for the entire DEM (slope calculation needs neighbors)
        risk_data = compute_risk(
            dem, rainfall=rainfall, duration=duration, 
            geology_factor=geology_factor, ndvi=None
        )

        # 3. Normalize risk data for coloring
        if risk_data.max() > risk_data.min():
            normalized_risk = (risk_data - risk_data.min()) / (risk_data.max() - risk_data.min())
        else:
            normalized_risk = np.zeros_like(risk_data)

        # 4. Apply colormap and create an RGBA image
        cmap = cm.jet
        colored_image_array = cmap(normalized_risk)

        # 5. Make everything outside the polygon transparent by setting alpha to 0
        colored_image_array[~mask, 3] = 0  # Set alpha channel to 0 where mask is False

        # 6. Convert to PIL Image and send back to the client
        img = Image.fromarray((colored_image_array * 255).astype(np.uint8))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr.seek(0)

        print("Dynamic polygon heatmap generated successfully.")
        return send_file(byte_arr, mimetype='image/png')

    except Exception as e:
        print(f"Error generating polygon heatmap: {e}")
        return make_response(jsonify({"error": f"Failed to generate polygon heatmap: {str(e)}"}), 500)

@app.route('/polygon_metrics', methods=['POST'])
def polygon_metrics():
    data = request.get_json()
    # Leaflet sends coordinates nested, so we access the first element.
    coordinates = data.get('coordinates', [[]])[0]

    if not coordinates or len(coordinates) < 3:
        return make_response(jsonify({"error": "No coordinates provided for polygon."}), 400)

    try:
        # --- Generate the same set of factors used for the heatmap ---
        rainfall = random.randint(0, 500)
        duration = random.randint(1, 72)
        geology_factor = random.uniform(0.1, 1.0)
        
        print(f"Calculating polygon metrics with factors: Rain={rainfall}, Duration={duration}, Geology={geology_factor:.2f}")

        # --- Perform the ACTUAL risk calculation ---
        # 1. Create the polygon mask
        mask = create_polygon_mask(dem.shape, coordinates, BOUNDS)

        # 2. Compute risk data
        risk_data = compute_risk(
            dem, rainfall=rainfall, duration=duration, 
            geology_factor=geology_factor, ndvi=None
        )

        # 3. Calculate metrics ONLY within the masked area
        risk_in_polygon = risk_data[mask]
        max_risk = risk_in_polygon.max() if risk_in_polygon.size > 0 else 0
        mean_risk = risk_in_polygon.mean() if risk_in_polygon.size > 0 else 0

        explanation = f"Analysis complete for selected area."

        return jsonify({
            "max_risk": round(float(max_risk), 2),
            "mean_risk": round(float(mean_risk), 2),
            "explanation": explanation,
            "factors": {
                "Rainfall (mm)": rainfall,
                "Duration (hrs)": duration,
                "Geology Instability": round(geology_factor, 2)
            }
        })
    except Exception as e:
        print(f"Error calculating polygon metrics: {e}")
        return make_response(jsonify({"error": "Failed to calculate metrics"}), 500)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
