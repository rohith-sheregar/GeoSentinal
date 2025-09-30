# dem_generator.py
# Creates a synthetic DEM and saves as dem.npy if none exists.

import numpy as np

def generate_dem(shape=(300, 300), hills=6, seed=1):
    np.random.seed(seed)
    x = np.linspace(-3, 3, shape[1])
    y = np.linspace(-3, 3, shape[0])
    X, Y = np.meshgrid(x, y)
    dem = np.zeros(shape, dtype=float)
    for _ in range(hills):
        cx = np.random.uniform(-2, 2)
        cy = np.random.uniform(-2, 2)
        amp = np.random.uniform(10, 80)
        sx = np.random.uniform(0.2, 1.0)
        sy = np.random.uniform(0.2, 1.0)
        dem += amp * np.exp(-(((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
    dem += np.linspace(0, 30, shape[0])[:, None]
    return dem

if __name__ == "__main__":
    dem = generate_dem()
    import numpy as _np
    _np.save("dem.npy", dem)
    print("Saved dem.npy shape", dem.shape)
