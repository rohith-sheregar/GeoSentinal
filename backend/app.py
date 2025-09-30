# app.py
from flask import Flask, send_file, jsonify, request, make_response
from flask_cors import CORS
import numpy as np
import io
from PIL import Image
from matplotlib import cm
import os

from risk import compute_risk

app = Flask(__name__)
CORS(app)

THIS_DIR = os.path.dirname(__file__)
DEM_PATH = os.path.join(THIS_DIR, "dem.npy")

if not os.path.exists(DEM_PATH):
    from dem_generator import generate_dem
    dem = generate_dem()
    np.save(DEM_PATH, dem)
else:
    dem = np.load(DEM_PATH)

BOUNDS = [12.95, 77.55, 13.05, 77.65]

@app.route("/risk_map.png")
def risk_map():
    try:
        rain = float(request.args.get("rain", "0"))
    except:
        rain = 0.0
    risk = compute_risk(dem, rainfall=rain)
    cmap = cm.get_cmap("hot")
    rgba = cmap(risk)
    alpha = np.clip((risk - 0.03) / (1.0 - 0.03), 0, 1)
    rgba[..., 3] = alpha
    rgba8 = (rgba * 255).astype("uint8")
    img = Image.fromarray(rgba8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = make_response(buf.getvalue())
    resp.headers.set("Content-Type", "image/png")
    return resp

@app.route("/metrics")
def metrics():
    try:
        rain = float(request.args.get("rain", "0"))
    except:
        rain = 0.0
    risk = compute_risk(dem, rainfall=rain)
    return jsonify(max_risk=float(risk.max()), mean_risk=float(risk.mean()))

@app.route("/bounds")
def bounds():
    s, w, n, e = BOUNDS
    return jsonify(south=s, west=w, north=n, east=e)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
