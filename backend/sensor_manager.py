import os
import json
import time
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import logging
from typing import Dict, List, Optional

class SensorManager:
    def __init__(self, config_path: str = None):
        self.sensor_data: Dict[str, queue.Queue] = {
            'displacement': queue.Queue(maxsize=1000),
            'strain': queue.Queue(maxsize=1000),
            'pore_pressure': queue.Queue(maxsize=1000),
            'rainfall': queue.Queue(maxsize=1000),
            'temperature': queue.Queue(maxsize=1000),
            'vibration': queue.Queue(maxsize=1000)
        }
        
        self.thresholds = {
            'displacement': 10.0,  # mm
            'strain': 0.002,      # strain ratio
            'pore_pressure': 50,  # kPa
            'rainfall': 100,      # mm/hr
            'temperature': 40,    # °C
            'vibration': 5.0      # m/s²
        }
        
        self.alert_callbacks = []
        self.is_running = False
        self.simulation_thread = None
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.thresholds.update(config.get('thresholds', {}))
    
    def start_monitoring(self):
        """Start the sensor monitoring system"""
        if not self.is_running:
            self.is_running = True
            self.simulation_thread = threading.Thread(target=self._simulate_sensor_data)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            logging.info("Sensor monitoring started")
    
    def stop_monitoring(self):
        """Stop the sensor monitoring system"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        logging.info("Sensor monitoring stopped")
    
    def register_alert_callback(self, callback):
        """Register a callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_readings(self) -> Dict[str, float]:
        """Get the most recent readings from all sensors"""
        current_readings = {}
        for sensor_type, sensor_queue in self.sensor_data.items():
            try:
                current_readings[sensor_type] = list(sensor_queue.queue)[-1]
            except IndexError:
                current_readings[sensor_type] = 0.0
        return current_readings
    
    def get_historical_data(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get historical sensor data for the specified time period"""
        historical_data = {}
        for sensor_type, sensor_queue in self.sensor_data.items():
            data_list = list(sensor_queue.queue)
            if len(data_list) > hours:
                historical_data[sensor_type] = data_list[-hours:]
            else:
                historical_data[sensor_type] = data_list
        return historical_data
    
    def _simulate_sensor_data(self):
        """Simulate sensor readings for testing and demonstration"""
        while self.is_running:
            timestamp = datetime.now()
            
            # Simulate realistic sensor readings with some randomness and trends
            readings = {
                'displacement': max(0, np.random.normal(5, 2)),  # mm
                'strain': max(0, np.random.normal(0.001, 0.0005)),  # ratio
                'pore_pressure': max(0, np.random.normal(30, 10)),  # kPa
                'rainfall': max(0, np.random.normal(50, 20)),  # mm/hr
                'temperature': np.random.normal(25, 5),  # °C
                'vibration': max(0, np.random.normal(2, 1))  # m/s²
            }
            
            # Add seasonal and time-of-day variations
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Temperature varies with time of day and season
            readings['temperature'] += 5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            readings['temperature'] += 10 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal cycle
            
            # Rainfall more likely in certain seasons and times
            if 0.1 * (1 + np.sin(2 * np.pi * day_of_year / 365)) > np.random.random():
                readings['rainfall'] *= 2
            
            # Add readings to queues
            for sensor_type, value in readings.items():
                if not self.sensor_data[sensor_type].full():
                    self.sensor_data[sensor_type].put(value)
                else:
                    # Remove oldest reading if queue is full
                    self.sensor_data[sensor_type].get()
                    self.sensor_data[sensor_type].put(value)
                
                # Check thresholds and trigger alerts
                if value > self.thresholds[sensor_type]:
                    self._trigger_alert(sensor_type, value)
            
            time.sleep(1)  # Update every second
    
    def _trigger_alert(self, sensor_type: str, value: float):
        """Trigger alert callbacks when thresholds are exceeded"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'sensor_type': sensor_type,
            'value': value,
            'threshold': self.thresholds[sensor_type],
            'message': f"WARNING: {sensor_type} exceeded threshold: {value:.2f}"
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def alert_handler(alert_data):
        print(f"ALERT: {alert_data['message']}")
    
    manager = SensorManager()
    manager.register_alert_callback(alert_handler)
    manager.start_monitoring()
    
    try:
        while True:
            time.sleep(5)
            current = manager.get_current_readings()
            print("Current readings:", current)
    except KeyboardInterrupt:
        manager.stop_monitoring()