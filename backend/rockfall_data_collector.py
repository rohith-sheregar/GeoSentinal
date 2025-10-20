import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import pickle
import pandas as pd
import cv2

class RockfallDataCollector:
    def __init__(self):
        self.data_dir = 'training_data'
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, 'dem'),
            os.path.join(self.data_dir, 'images'),
            os.path.join(self.data_dir, 'sensor_data'),
            os.path.join(self.data_dir, 'weather')
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def collect_geological_data(self, location: Dict[str, float]) -> Dict:
        """
        Collect geological data for a given location.
        Location should be a dict with 'lat' and 'lon' keys.
        
        Data Sources:
        1. NASA SRTM DEM data: https://www2.jpl.nasa.gov/srtm/
        2. USGS Earth Explorer: https://earthexplorer.usgs.gov/
        3. OpenTopography: https://opentopography.org/
        """
        filename = os.path.join(
            self.data_dir, 
            'dem', 
            f"dem_{location['lat']}_{location['lon']}.npy"
        )
        
        if os.path.exists(filename):
            return np.load(filename)
            
        # For development/testing, generate synthetic DEM
        dem_size = (64, 64)
        x = np.linspace(-1, 1, dem_size[0])
        y = np.linspace(-1, 1, dem_size[1])
        X, Y = np.meshgrid(x, y)
        
        # Generate realistic terrain features
        dem = np.zeros(dem_size)
        
        # Add base elevation
        dem += 100 + 50 * np.sin(3 * X) * np.cos(3 * Y)
        
        # Add random peaks
        n_peaks = np.random.randint(3, 7)
        for _ in range(n_peaks):
            px = np.random.uniform(-1, 1)
            py = np.random.uniform(-1, 1)
            height = np.random.uniform(20, 100)
            dem += height * np.exp(-5 * ((X - px)**2 + (Y - py)**2))
        
        # Add noise for texture
        dem += np.random.normal(0, 2, dem_size)
        
        # Save DEM data
        np.save(filename, dem)
        return dem

    def collect_weather_data(self, location: Dict[str, float]) -> Dict:
        """
        Collect weather data using OpenWeatherMap API or local weather station data.
        You'll need to sign up for an API key at: https://openweathermap.org/api
        """
        api_key = os.getenv('OPENWEATHER_API_KEY')
        cache_file = os.path.join(
            self.data_dir,
            'weather',
            f"weather_{location['lat']}_{location['lon']}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        if api_key:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'appid': api_key,
                    'units': 'metric'
                }
                response = requests.get(url, params=params)
                data = response.json()
                
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'rainfall': data.get('rain', {}).get('1h', 0),
                    'wind_speed': data['wind']['speed']
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(weather_data, f)
                    
                return weather_data
                
            except Exception as e:
                logging.warning(f"Failed to fetch weather data: {e}")
        
        # Generate synthetic weather data if API fails or key not available
        return self._generate_synthetic_weather()

    def _generate_synthetic_weather(self) -> Dict:
        """Generate realistic synthetic weather data"""
        hour = datetime.now().hour
        day_of_year = datetime.now().timetuple().tm_yday
        
        # Temperature varies with time of day and season
        base_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)
        temp = base_temp + 10 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Rainfall probability increases with humidity
        humidity = np.random.normal(70, 10)
        rainfall_prob = (humidity - 50) / 100
        rainfall = np.random.exponential(5) if np.random.random() < rainfall_prob else 0
        
        return {
            'temperature': temp,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': np.random.normal(5, 2)
        }

    def collect_sensor_data(self, duration_hours: int = 24) -> pd.DataFrame:
        """
        Simulate sensor data collection from various monitoring devices.
        In real deployment, this would interface with actual sensors.
        
        Types of sensors typically used:
        1. Extensometers - for displacement
        2. Inclinometers - for tilt
        3. Piezometers - for groundwater pressure
        4. Strain gauges - for deformation
        5. Seismometers - for vibration
        """
        now = datetime.now()
        timestamps = [
            now - timedelta(hours=i)
            for i in range(duration_hours)
        ]
        
        data = {
            'timestamp': timestamps,
            'displacement_mm': [],
            'tilt_degrees': [],
            'pore_pressure_kpa': [],
            'strain_ratio': [],
            'vibration_mps2': []
        }
        
        # Generate realistic time series with trends and noise
        for _ in range(duration_hours):
            # Base values
            displacement = np.random.normal(5, 1)
            tilt = np.random.normal(2, 0.5)
            pressure = np.random.normal(30, 5)
            strain = np.random.normal(0.001, 0.0002)
            vibration = np.random.normal(0.5, 0.1)
            
            # Add sensor drift
            drift = np.random.normal(0, 0.1)
            displacement += drift
            tilt += drift * 0.2
            
            # Weather effects
            weather = self._generate_synthetic_weather()
            if weather['rainfall'] > 5:
                pressure *= 1.2
                displacement *= 1.1
            
            data['displacement_mm'].append(max(0, displacement))
            data['tilt_degrees'].append(tilt)
            data['pore_pressure_kpa'].append(max(0, pressure))
            data['strain_ratio'].append(max(0, strain))
            data['vibration_mps2'].append(max(0, vibration))
        
        return pd.DataFrame(data)

    def collect_image_data(self, location: Dict[str, float]) -> np.ndarray:
        """
        Collect or simulate drone/satellite imagery.
        In real deployment, this would interface with:
        1. Drone cameras
        2. Satellite imagery APIs
        3. Ground-based cameras
        """
        filename = os.path.join(
            self.data_dir,
            'images',
            f"image_{location['lat']}_{location['lon']}.npy"
        )
        
        if os.path.exists(filename):
            return np.load(filename)
        
        # Generate synthetic image data (64x64x3 RGB)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Add terrain texture
        noise = np.random.normal(0, 1, (64, 64))
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
        
        # Create different channels for varied appearance
        image[:,:,0] = noise  # Red channel
        image[:,:,1] = np.roll(noise, 10)  # Green channel
        image[:,:,2] = np.roll(noise, -10)  # Blue channel
        
        # Add some random features
        for _ in range(5):
            x = np.random.randint(0, 64)
            y = np.random.randint(0, 64)
            radius = np.random.randint(3, 8)
            color = np.random.randint(0, 255, 3)
            cv2.circle(image, (x, y), radius, color.tolist(), -1)
        
        # Save image
        np.save(filename, image)
        return image

    def prepare_training_data(self, locations: List[Dict[str, float]], historical_days: int = 30) -> Dict:
        """
        Prepare a complete training dataset combining all data sources.
        Returns a dictionary containing features and labels.
        """
        training_data = {
            'dem_data': [],
            'image_data': [],
            'weather_data': [],
            'sensor_data': [],
            'labels': []
        }
        
        for location in locations:
            # Collect all data sources
            dem = self.collect_geological_data(location)
            image = self.collect_image_data(location)
            weather = self.collect_weather_data(location)
            sensors = self.collect_sensor_data(24)  # Last 24 hours
            
            # Calculate risk score (synthetic for training)
            risk_score = self._calculate_risk_score(dem, weather, sensors)
            
            # Store data
            training_data['dem_data'].append(dem)
            training_data['image_data'].append(image)
            training_data['weather_data'].append(weather)
            training_data['sensor_data'].append(sensors.to_dict('records'))
            training_data['labels'].append(risk_score)
        
        return training_data

    def _calculate_risk_score(self, dem: np.ndarray, weather: Dict, sensors: pd.DataFrame) -> float:
        """Calculate synthetic risk score for training data"""
        # Slope-based risk
        gradient = np.gradient(dem)
        slope_risk = np.max(np.hypot(gradient[0], gradient[1])) / np.pi
        
        # Weather-based risk
        weather_risk = 0.0
        if weather['rainfall'] > 10:
            weather_risk += 0.3
        if weather['temperature'] < 0:
            weather_risk += 0.2
        
        # Sensor-based risk
        sensor_risk = 0.0
        recent_sensors = sensors.iloc[-1]
        if recent_sensors['displacement_mm'] > 8:
            sensor_risk += 0.3
        if recent_sensors['pore_pressure_kpa'] > 40:
            sensor_risk += 0.2
        
        # Combine risks with weights
        total_risk = (0.4 * slope_risk + 
                     0.3 * weather_risk + 
                     0.3 * sensor_risk)
        
        # Add some randomness for training diversity
        total_risk = np.clip(total_risk + np.random.normal(0, 0.1), 0, 1)
        
        return float(total_risk)

    def save_dataset(self, data: Dict, filename: str):
        """Save the collected dataset"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_dataset(self, filename: str) -> Dict:
        """Load a saved dataset"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Example usage
    collector = RockfallDataCollector()
    
    # Generate sample locations (around a central point)
    center_lat, center_lon = 13.0, 77.6  # Example: Bangalore
    locations = [
        {
            'lat': center_lat + np.random.normal(0, 0.1),
            'lon': center_lon + np.random.normal(0, 0.1)
        }
        for _ in range(5)
    ]
    
    # Collect training data
    print("Collecting training data...")
    training_data = collector.prepare_training_data(locations)
    
    # Save the dataset
    collector.save_dataset(training_data, 'rockfall_training_data.pkl')
    print("Dataset saved successfully!")