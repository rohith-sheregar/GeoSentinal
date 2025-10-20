import numpy as np
import requests
import rasterio
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple
import json
import logging

class DataCollector:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.cache_dir = 'data_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'dem': {
                'source': 'SRTM',
                'resolution': 30,  # meters
                'bounds': {
                    'north': None,
                    'south': None,
                    'east': None,
                    'west': None
                }
            },
            'weather_api': {
                'provider': 'openweathermap',
                'api_key': os.getenv('OPENWEATHER_API_KEY')
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config

    def get_dem_data(self, lat: float, lon: float, size: int = 64) -> np.ndarray:
        """
        Fetch DEM data for a given location.
        For demo/development, generates synthetic DEM if no real data available.
        """
        cache_file = os.path.join(self.cache_dir, f'dem_{lat}_{lon}_{size}.npy')
        
        if os.path.exists(cache_file):
            return np.load(cache_file)
            
        try:
            # Try to fetch real DEM data
            if self.config['dem']['source'] == 'SRTM':
                dem_data = self._fetch_srtm_data(lat, lon, size)
            else:
                dem_data = self._generate_synthetic_dem(size)
        except Exception as e:
            logging.warning(f"Failed to fetch real DEM data: {e}")
            dem_data = self._generate_synthetic_dem(size)
            
        np.save(cache_file, dem_data)
        return dem_data

    def _generate_synthetic_dem(self, size: int) -> np.ndarray:
        """Generate synthetic DEM data for testing"""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate base terrain
        Z = (np.sin(5*X) * np.cos(5*Y) + 
             np.random.normal(0, 0.1, (size, size)))
        
        # Add some peaks and valleys
        peaks = np.random.randint(3, 7)
        for _ in range(peaks):
            px = np.random.uniform(-1, 1)
            py = np.random.uniform(-1, 1)
            height = np.random.uniform(0.5, 2.0)
            Z += height * np.exp(-5*((X-px)**2 + (Y-py)**2))
        
        return Z

    def get_weather_data(self, lat: float, lon: float) -> Dict:
        """Fetch current weather data"""
        cache_file = os.path.join(self.cache_dir, 
                                f'weather_{lat}_{lon}_{datetime.now().strftime("%Y%m%d")}.json')
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
                
        try:
            if self.config['weather_api']['provider'] == 'openweathermap':
                api_key = self.config['weather_api']['api_key']
                url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
                response = requests.get(url)
                data = response.json()
                
                weather_data = {
                    'temperature': data['main']['temp'] - 273.15,  # Convert to Celsius
                    'rainfall': data['rain']['1h'] if 'rain' in data else 0,
                    'humidity': data['main']['humidity']
                }
            else:
                weather_data = self._generate_synthetic_weather()
        except Exception as e:
            logging.warning(f"Failed to fetch real weather data: {e}")
            weather_data = self._generate_synthetic_weather()
            
        with open(cache_file, 'w') as f:
            json.dump(weather_data, f)
            
        return weather_data

    def _generate_synthetic_weather(self) -> Dict:
        """Generate synthetic weather data for testing"""
        hour = datetime.now().hour
        day_of_year = datetime.now().timetuple().tm_yday
        
        # Temperature varies with time of day and season
        base_temp = 25 + 5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
        temp = base_temp + 10 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal cycle
        
        # Rainfall more likely in certain seasons
        rainfall_prob = 0.1 * (1 + np.sin(2 * np.pi * day_of_year / 365))
        rainfall = np.random.exponential(5) if np.random.random() < rainfall_prob else 0
        
        return {
            'temperature': temp,
            'rainfall': rainfall,
            'humidity': np.random.uniform(40, 90)
        }

    def get_sensor_data(self) -> Dict:
        """
        Simulate geotechnical sensor readings.
        In production, this would interface with real sensors.
        """
        # Base values
        data = {
            'displacement': np.random.normal(5, 1),    # mm
            'strain': np.random.normal(0.001, 0.0002), # ratio
            'pore_pressure': np.random.normal(30, 5),  # kPa
            'vibration': np.random.normal(2, 0.5)      # m/s²
        }
        
        # Add time-based variations
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Daytime
            data['vibration'] *= 1.5  # More activity during day
            
        # Add weather-based effects
        weather = self.get_weather_data(0, 0)  # Using default location
        if weather['rainfall'] > 5:  # Heavy rain
            data['pore_pressure'] *= 1.2
            data['displacement'] *= 1.1
            
        return data

    def get_training_data(self, size: int = 1000) -> Tuple[Dict, np.ndarray]:
        """
        Generate training data for the ML models.
        Returns features and labels for training.
        """
        features = {
            'dem_data': [],
            'weather_data': [],
            'sensor_data': [],
            'labels': []
        }
        
        for _ in range(size):
            # Generate synthetic data point
            lat = np.random.uniform(-1, 1)
            lon = np.random.uniform(-1, 1)
            
            dem = self.get_dem_data(lat, lon)
            weather = self.get_weather_data(lat, lon)
            sensors = self.get_sensor_data()
            
            # Generate label (probability of rockfall)
            # This is a simplified model for synthetic data
            prob_rockfall = 0.0
            
            # Slope-based probability
            slope = np.max(np.gradient(dem))
            prob_rockfall += 0.3 * (slope / np.pi)
            
            # Weather-based probability
            if weather['rainfall'] > 10:
                prob_rockfall += 0.2
            
            # Sensor-based probability
            if sensors['displacement'] > 8:
                prob_rockfall += 0.3
            if sensors['pore_pressure'] > 40:
                prob_rockfall += 0.2
                
            # Add some noise
            prob_rockfall = np.clip(prob_rockfall + np.random.normal(0, 0.1), 0, 1)
            
            features['dem_data'].append(dem)
            features['weather_data'].append(weather)
            features['sensor_data'].append(sensors)
            features['labels'].append(prob_rockfall)
        
        return features, np.array(features['labels'])

    def save_data(self, data: Dict, filename: str):
        """Save collected data to file"""
        filepath = os.path.join(self.cache_dir, filename)
        np.savez(filepath, **data)

    def load_data(self, filename: str) -> Dict:
        """Load collected data from file"""
        filepath = os.path.join(self.cache_dir, filename)
        with np.load(filepath) as data:
            return dict(data)

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Get DEM data
    dem = collector.get_dem_data(0, 0)
    print("DEM shape:", dem.shape)
    
    # Get weather data
    weather = collector.get_weather_data(0, 0)
    print("Weather:", weather)
    
    # Get sensor data
    sensors = collector.get_sensor_data()
    print("Sensor readings:", sensors)
    
    # Generate training data
    features, labels = collector.get_training_data(size=10)
    print("Generated training data size:", len(labels))