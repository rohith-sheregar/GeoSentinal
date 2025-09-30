# risk.py
# Slope and risk computation functions.

from skimage.draw import polygon
import numpy as np

def norm(a):
    a = np.array(a, dtype=float)
    mn = a.min()
    mx = a.max()
    if mx - mn < 1e-9:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def compute_slope(dem, dx=1.0, dy=1.0):
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.hypot(gx, gy)
    return slope

def compute_risk(dem, rainfall=0.0, duration=0.0, ndvi=None, geology_factor=0.5, weights=(0.4, 0.25, 0.15, 0.1, 0.1)):
    """
    Computes a risk score based on multiple environmental factors.
    Weights correspond to: slope, rainfall, duration, ndvi, geology.
    """
    # Normalize slope from 0 to 1
    s = norm(compute_slope(dem))

    # Normalize rainfall against its expected range (0-500) and create an array
    r = np.full_like(s, rainfall / 500.0)

    # Normalize duration against its expected range (e.g., 1-72 hours)
    d = np.full_like(s, duration / 72.0)

    # Normalize NDVI (vegetation cover)
    v = norm(ndvi) if ndvi is not None else np.zeros_like(s)

    w_s, w_r, w_d, w_v, w_g = weights
    risk = w_s * s + w_r * r + w_d * d + w_g * geology_factor + w_v * (1 - v)
    return risk

def create_polygon_mask(shape, coordinates, bounds):
    """
    Creates a boolean mask for a DEM array from a list of geographic coordinates.

    Args:
        shape (tuple): The (height, width) of the DEM array.
        coordinates (list): A list of dictionaries, e.g., [{'lat': y, 'lng': x}, ...].
        bounds (list): The geographic bounds [south, west, north, east] of the DEM.

    Returns:
        np.ndarray: A boolean array of the given shape, True inside the polygon.
    """
    s, w, n, e = bounds
    height, width = shape

    # Convert lat/lng to pixel coordinates
    pixel_coords = []
    for coord in coordinates:
        lat, lng = coord['lat'], coord['lng']
        # Clamp coordinates to bounds to avoid errors
        lat = max(s, min(n, lat))
        lng = max(w, min(e, lng))
        # Convert geo coordinates to pixel indices
        row = (n - lat) / (n - s) * (height - 1)
        col = (lng - w) / (e - w) * (width - 1)
        pixel_coords.append((row, col))

    # Create an empty mask and draw the polygon on it
    mask = np.zeros(shape, dtype=bool)
    rows, cols = zip(*pixel_coords)
    rr, cc = polygon(rows, cols, shape)
    mask[rr, cc] = True
    return mask
